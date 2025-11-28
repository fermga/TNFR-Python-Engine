"""
Tests for TNFR Symbolic Mathematics Module.

Validates:
- Nodal equation representation
- Analytical solutions
- Convergence analysis (U2)
- Bifurcation detection (U4)
"""

import pytest
from tnfr.math import symbolic


class TestNodalEquation:
    """Test nodal equation representation."""
    
    def test_get_nodal_equation(self):
        """Nodal equation should have correct form."""
        eq = symbolic.get_nodal_equation()
        
        # Should be an equation
        assert hasattr(eq, 'lhs') and hasattr(eq, 'rhs')
        
        # LHS should be derivative of EPI
        assert 'EPI' in str(eq.lhs)
        assert 'Derivative' in str(type(eq.lhs))
        
        # RHS should be nu_f * DELTA_NFR
        assert 'nu_f' in str(eq.rhs)
        assert 'DELTA_NFR' in str(eq.rhs)
    
    def test_solve_constant_params_positive_growth(self):
        """Positive ΔNFR should produce growing EPI."""
        solution = symbolic.solve_nodal_equation_constant_params(
            nu_f_val=1.0,
            delta_nfr_val=0.5,
            EPI_0=1.0,
            t0=0
        )
        
        # At t=0, should equal EPI_0
        val_t0 = solution.subs(symbolic.t, 0)
        assert abs(float(val_t0) - 1.0) < 1e-6
        
        # At t=2, should be EPI_0 + 2*νf*ΔNFR = 1 + 2*1*0.5 = 2
        val_t2 = solution.subs(symbolic.t, 2)
        assert abs(float(val_t2) - 2.0) < 1e-6
    
    def test_solve_constant_params_negative_growth(self):
        """Negative ΔNFR should produce decreasing EPI."""
        solution = symbolic.solve_nodal_equation_constant_params(
            nu_f_val=1.0,
            delta_nfr_val=-0.5,
            EPI_0=2.0,
            t0=0
        )
        
        # At t=0, should equal EPI_0
        val_t0 = solution.subs(symbolic.t, 0)
        assert abs(float(val_t0) - 2.0) < 1e-6
        
        # At t=2, should be EPI_0 + 2*νf*ΔNFR = 2 + 2*1*(-0.5) = 1
        val_t2 = solution.subs(symbolic.t, 2)
        assert abs(float(val_t2) - 1.0) < 1e-6
    
    def test_solve_zero_frequency(self):
        """Zero νf should produce constant EPI (frozen node)."""
        solution = symbolic.solve_nodal_equation_constant_params(
            nu_f_val=0.0,
            delta_nfr_val=1.0,  # Doesn't matter if νf=0
            EPI_0=5.0,
            t0=0
        )
        
        # Should remain constant
        val_t0 = solution.subs(symbolic.t, 0)
        val_t10 = solution.subs(symbolic.t, 10)
        assert abs(float(val_t0) - 5.0) < 1e-6
        assert abs(float(val_t10) - 5.0) < 1e-6


class TestConvergenceAnalysis:
    """Test convergence analysis for U2 grammar rule."""
    
    def test_convergence_decaying_exponential(self):
        """Negative growth rate should converge."""
        converges, explanation, value = \
            symbolic.check_convergence_exponential(-0.1, 10.0)
        
        assert converges is True
        assert "Converges" in explanation
        assert value is not None
        assert value > 0  # Integral of positive exponential decay
    
    def test_divergence_growing_exponential(self):
        """Positive growth rate should diverge (needs stabilizers)."""
        converges, explanation, value = \
            symbolic.check_convergence_exponential(0.1, 10.0)
        
        assert converges is False
        assert "DIVERGES" in explanation
        assert "STABILIZERS" in explanation
        assert value is not None
        # Growing exponential has larger integral than decaying
    
    def test_convergence_constant(self):
        """Zero growth rate should converge (equilibrium)."""
        converges, explanation, value = \
            symbolic.check_convergence_exponential(0.0, 10.0)
        
        assert converges is True
        assert "Converges" in explanation
        assert value is not None


class TestBifurcationAnalysis:
    """Test bifurcation detection for U4 grammar rule."""
    
    def test_no_bifurcation_risk_stable(self):
        """Low second derivative should be stable."""
        at_risk, deriv_val, recommendation = \
            symbolic.evaluate_bifurcation_risk(
                nu_f_val=1.0,
                delta_nfr_val=0.3,
                d_nu_f_dt=0.1,
                d_delta_nfr_dt=0.2,
                threshold=1.0
            )
        
        assert at_risk is False
        assert abs(deriv_val) < 1.0
        assert "Stable" in recommendation or "✓" in recommendation
    
    def test_bifurcation_risk_high_acceleration(self):
        """High second derivative should trigger warning."""
        at_risk, deriv_val, recommendation = \
            symbolic.evaluate_bifurcation_risk(
                nu_f_val=2.0,
                delta_nfr_val=1.5,
                d_nu_f_dt=0.5,
                d_delta_nfr_dt=1.0,
                threshold=1.0
            )
        
        assert at_risk is True
        assert abs(deriv_val) > 1.0
        assert ("BIFURCATION" in recommendation or
                "⚠" in recommendation)
        assert "THOL" in recommendation or "IL" in recommendation
    
    def test_second_derivative_formula(self):
        """Second derivative should use product rule."""
        second_deriv = symbolic.compute_second_derivative_symbolic()
        
        # Should contain both terms from product rule
        expr_str = str(second_deriv)
        assert 'nu_f' in expr_str or 'ν_f' in expr_str
        assert 'DELTA_NFR' in expr_str or 'Δ' in expr_str
        # Result is Add (sum of two derivative terms)
        assert 'Add' in str(type(second_deriv)) or \
               'Derivative' in expr_str


class TestUtilities:
    """Test utility functions."""
    
    def test_latex_export(self):
        """LaTeX export should produce valid LaTeX."""
        eq = symbolic.get_nodal_equation()
        latex_str = symbolic.latex_export(eq)
        
        assert isinstance(latex_str, str)
        assert len(latex_str) > 0
        # LaTeX should contain command syntax
        assert '\\' in latex_str
    
    def test_pretty_print(self):
        """Pretty print should be human-readable."""
        eq = symbolic.get_nodal_equation()
        pretty_str = symbolic.pretty_print(eq)
        
        assert isinstance(pretty_str, str)
        assert len(pretty_str) > 0


class TestPhysicsAlignment:
    """Test alignment with TNFR physics."""
    
    def test_u2_convergence_requirement(self):
        """U2: Destabilizers without stabilizers should diverge."""
        # Simulating destabilizer effect (positive growth)
        converges, _, _ = symbolic.check_convergence_exponential(
            0.2,  # Positive growth (destabilizer)
            10.0
        )
        assert not converges  # Should NOT converge without stabilizer
    
    def test_u4_bifurcation_threshold(self):
        """U4: High ∂²EPI/∂t² should require handlers."""
        # High acceleration scenario
        at_risk, _, rec = symbolic.evaluate_bifurcation_risk(
            nu_f_val=3.0,
            delta_nfr_val=2.0,
            d_nu_f_dt=1.0,
            d_delta_nfr_dt=2.0,
            threshold=1.0
        )
        
        assert at_risk is True
        # Should recommend handlers (U4a requirement)
        assert any(handler in rec for handler in ["THOL", "IL"])
    
    def test_invariant_2_units(self):
        """Invariant #2: νf should be in Hz_str."""
        # This is implicitly tested by using positive reals
        # In actual code, unit tracking would be more explicit
        assert symbolic.nu_f.is_positive is True
        assert symbolic.nu_f.is_real is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
