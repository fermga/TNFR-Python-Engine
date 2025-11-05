"""Tests for canonical TNFR nodal equation implementation.

This test suite validates the explicit implementation of the fundamental
TNFR equation: ∂EPI/∂t = νf · ΔNFR(t)

Test coverage:
  1. Basic equation computation
  2. Unit validation (Hz_str)
  3. Edge cases (zero values, negative ΔNFR)
  4. Error handling (invalid inputs)
  5. Integration with existing code
  6. TNFR invariant preservation
"""

from __future__ import annotations

import math

import pytest

from tnfr.dynamics.canonical import (
    NodalEquationResult,
    compute_canonical_nodal_derivative,
    validate_nodal_gradient,
    validate_structural_frequency,
)


class TestCanonicalNodalEquation:
    """Test the canonical TNFR nodal equation: ∂EPI/∂t = νf · ΔNFR(t)."""

    def test_basic_computation(self):
        """Verify basic equation computation matches νf · ΔNFR."""
        result = compute_canonical_nodal_derivative(
            nu_f=1.5,
            delta_nfr=0.4,
            validate_units=False,
        )
        
        assert result.derivative == pytest.approx(1.5 * 0.4)
        assert result.nu_f == 1.5
        assert result.delta_nfr == 0.4
        assert result.validated is False

    def test_computation_with_validation(self):
        """Verify computation with unit validation enabled."""
        result = compute_canonical_nodal_derivative(
            nu_f=2.0,
            delta_nfr=-0.5,
            validate_units=True,
        )
        
        assert result.derivative == pytest.approx(-1.0)
        assert result.validated is True

    def test_zero_frequency_yields_zero_derivative(self):
        """νf=0 represents structural silence and should yield zero derivative."""
        result = compute_canonical_nodal_derivative(
            nu_f=0.0,
            delta_nfr=0.8,
            validate_units=True,
        )
        
        assert result.derivative == pytest.approx(0.0)
        assert result.nu_f == 0.0

    def test_zero_gradient_yields_zero_derivative(self):
        """ΔNFR=0 represents equilibrium and should yield zero derivative."""
        result = compute_canonical_nodal_derivative(
            nu_f=1.2,
            delta_nfr=0.0,
            validate_units=True,
        )
        
        assert result.derivative == pytest.approx(0.0)
        assert result.delta_nfr == 0.0

    def test_negative_gradient_produces_negative_derivative(self):
        """Negative ΔNFR should produce negative derivative (contraction)."""
        result = compute_canonical_nodal_derivative(
            nu_f=1.0,
            delta_nfr=-0.3,
            validate_units=True,
        )
        
        assert result.derivative == pytest.approx(-0.3)
        assert result.derivative < 0

    def test_positive_gradient_produces_positive_derivative(self):
        """Positive ΔNFR should produce positive derivative (expansion)."""
        result = compute_canonical_nodal_derivative(
            nu_f=1.0,
            delta_nfr=0.3,
            validate_units=True,
        )
        
        assert result.derivative == pytest.approx(0.3)
        assert result.derivative > 0

    def test_large_values_are_handled_correctly(self):
        """Test computation with large but valid values."""
        result = compute_canonical_nodal_derivative(
            nu_f=100.0,
            delta_nfr=50.0,
            validate_units=True,
        )
        
        assert result.derivative == pytest.approx(5000.0)

    def test_small_values_preserve_precision(self):
        """Test computation with small values preserves numerical precision."""
        result = compute_canonical_nodal_derivative(
            nu_f=0.001,
            delta_nfr=0.002,
            validate_units=True,
        )
        
        assert result.derivative == pytest.approx(0.000002)


class TestStructuralFrequencyValidation:
    """Test validation of structural frequency (νf) in Hz_str units."""

    def test_accepts_positive_frequency(self):
        """Positive frequencies should be accepted."""
        validated = validate_structural_frequency(1.5)
        assert validated == 1.5

    def test_accepts_zero_frequency(self):
        """Zero frequency (structural silence) should be accepted."""
        validated = validate_structural_frequency(0.0)
        assert validated == 0.0

    def test_rejects_negative_frequency(self):
        """Negative frequencies violate TNFR physics."""
        with pytest.raises(ValueError, match="non-negative"):
            validate_structural_frequency(-1.0)

    def test_rejects_nan(self):
        """NaN is not a valid structural frequency."""
        with pytest.raises(ValueError, match="finite"):
            validate_structural_frequency(float("nan"))

    def test_rejects_infinity(self):
        """Infinite frequency is not physically meaningful."""
        with pytest.raises(ValueError, match="finite"):
            validate_structural_frequency(float("inf"))
        
        with pytest.raises(ValueError, match="finite"):
            validate_structural_frequency(float("-inf"))

    def test_rejects_non_numeric(self):
        """Non-numeric inputs should be rejected."""
        with pytest.raises(TypeError, match="numeric"):
            validate_structural_frequency("invalid")  # type: ignore
        
        with pytest.raises(TypeError, match="numeric"):
            validate_structural_frequency(None)  # type: ignore

    def test_accepts_numeric_strings(self):
        """Numeric strings are coerced to float (Python float() behavior).
        
        Note: The function accepts any input that Python's float() can convert,
        including strings. While the type annotation says float, this follows
        Python's duck typing convention where float() performs conversion.
        """
        validated = validate_structural_frequency("1.5")  # type: ignore
        assert validated == 1.5
        assert isinstance(validated, float)

    def test_coerces_integer_to_float(self):
        """Integer inputs should be coerced to float."""
        validated = validate_structural_frequency(5)
        assert validated == 5.0
        assert isinstance(validated, float)


class TestNodalGradientValidation:
    """Test validation of nodal gradient (ΔNFR) operator."""

    def test_accepts_positive_gradient(self):
        """Positive ΔNFR (expansion) should be accepted."""
        validated = validate_nodal_gradient(0.5)
        assert validated == 0.5

    def test_accepts_negative_gradient(self):
        """Negative ΔNFR (contraction) should be accepted."""
        validated = validate_nodal_gradient(-0.5)
        assert validated == -0.5

    def test_accepts_zero_gradient(self):
        """Zero ΔNFR (equilibrium) should be accepted."""
        validated = validate_nodal_gradient(0.0)
        assert validated == 0.0

    def test_rejects_nan(self):
        """NaN is not a valid nodal gradient."""
        with pytest.raises(ValueError, match="finite"):
            validate_nodal_gradient(float("nan"))

    def test_rejects_infinity(self):
        """Infinite gradient is not physically meaningful."""
        with pytest.raises(ValueError, match="finite"):
            validate_nodal_gradient(float("inf"))
        
        with pytest.raises(ValueError, match="finite"):
            validate_nodal_gradient(float("-inf"))

    def test_rejects_non_numeric(self):
        """Non-numeric inputs should be rejected."""
        with pytest.raises(TypeError, match="numeric"):
            validate_nodal_gradient("invalid")  # type: ignore
        
        with pytest.raises(TypeError, match="numeric"):
            validate_nodal_gradient(None)  # type: ignore

    def test_accepts_numeric_strings(self):
        """Numeric strings are coerced to float (Python float() behavior).
        
        Note: The function accepts any input that Python's float() can convert,
        including strings. While the type annotation says float, this follows
        Python's duck typing convention where float() performs conversion.
        """
        validated = validate_nodal_gradient("0.5")  # type: ignore
        assert validated == 0.5
        assert isinstance(validated, float)

    def test_coerces_integer_to_float(self):
        """Integer inputs should be coerced to float."""
        validated = validate_nodal_gradient(-3)
        assert validated == -3.0
        assert isinstance(validated, float)


class TestNodalEquationResult:
    """Test the NodalEquationResult named tuple."""

    def test_result_structure(self):
        """Verify result contains expected fields."""
        result = NodalEquationResult(
            derivative=1.5,
            nu_f=2.0,
            delta_nfr=0.75,
            validated=True,
        )
        
        assert result.derivative == 1.5
        assert result.nu_f == 2.0
        assert result.delta_nfr == 0.75
        assert result.validated is True

    def test_result_is_immutable(self):
        """Result should be immutable (NamedTuple property)."""
        result = NodalEquationResult(1.0, 2.0, 0.5, True)
        
        with pytest.raises(AttributeError):
            result.derivative = 2.0  # type: ignore


class TestCanonicalEquationInvariants:
    """Test TNFR invariants specified in AGENTS.md."""

    def test_operator_closure_preserved(self):
        """Verify operator composition yields valid states."""
        # Apply equation twice (composition)
        result1 = compute_canonical_nodal_derivative(1.0, 0.5)
        result2 = compute_canonical_nodal_derivative(1.0, result1.derivative)
        
        # Both should be valid
        assert isinstance(result1.derivative, float)
        assert isinstance(result2.derivative, float)
        assert math.isfinite(result1.derivative)
        assert math.isfinite(result2.derivative)

    def test_zero_frequency_implies_silence(self):
        """νf=0 should freeze evolution (silence operator)."""
        result = compute_canonical_nodal_derivative(0.0, 0.5)
        assert result.derivative == 0.0
        
        result = compute_canonical_nodal_derivative(0.0, -0.8)
        assert result.derivative == 0.0

    def test_zero_gradient_implies_equilibrium(self):
        """ΔNFR=0 should halt reorganization."""
        result = compute_canonical_nodal_derivative(1.5, 0.0)
        assert result.derivative == 0.0

    def test_sign_of_gradient_controls_direction(self):
        """ΔNFR sign should determine expansion vs contraction."""
        expansion = compute_canonical_nodal_derivative(1.0, 0.5)
        contraction = compute_canonical_nodal_derivative(1.0, -0.5)
        
        assert expansion.derivative > 0  # Expansion
        assert contraction.derivative < 0  # Contraction
        assert abs(expansion.derivative) == abs(contraction.derivative)

    def test_magnitude_scales_linearly(self):
        """Doubling νf or ΔNFR should double the derivative."""
        base = compute_canonical_nodal_derivative(1.0, 0.5)
        scaled_freq = compute_canonical_nodal_derivative(2.0, 0.5)
        scaled_grad = compute_canonical_nodal_derivative(1.0, 1.0)
        
        assert scaled_freq.derivative == pytest.approx(2 * base.derivative)
        assert scaled_grad.derivative == pytest.approx(2 * base.derivative)


class TestIntegrationWithExistingCode:
    """Test that canonical implementation is consistent with existing integrators."""

    def test_matches_integrator_computation(self):
        """Verify canonical function matches integrators.py line 321."""
        # This is the computation at line 321 in integrators.py: base = vf * dnfr
        vf = 1.5
        dnfr = 0.4
        expected = vf * dnfr
        
        result = compute_canonical_nodal_derivative(vf, dnfr, validate_units=False)
        
        assert result.derivative == pytest.approx(expected)

    def test_canonical_function_is_drop_in_replacement(self):
        """Canonical function can replace inline multiplication."""
        # Values from typical TNFR node
        test_cases = [
            (1.0, 0.5),
            (2.3, -0.7),
            (0.0, 1.0),
            (1.5, 0.0),
            (0.8, -0.3),
        ]
        
        for vf, dnfr in test_cases:
            inline_result = vf * dnfr
            canonical_result = compute_canonical_nodal_derivative(
                vf, dnfr, validate_units=False
            )
            assert canonical_result.derivative == pytest.approx(inline_result)
