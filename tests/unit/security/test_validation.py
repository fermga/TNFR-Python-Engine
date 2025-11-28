"""Tests for TNFR structural data validation.

These tests verify that input validation functions properly enforce
TNFR structural invariants and prevent invalid data from being persisted.
"""

from __future__ import annotations

import math

import pytest

from tnfr.security.validation import (
    validate_coherence_value,
    validate_nodal_input,
    validate_phase_value,
    validate_sense_index,
    validate_structural_frequency,
)


class TestValidateStructuralFrequency:
    """Tests for structural frequency (νf) validation."""

    def test_valid_frequencies(self) -> None:
        """Test that valid frequencies pass validation."""
        assert validate_structural_frequency(0.5) == 0.5
        assert validate_structural_frequency(0.0) == 0.0  # Silence operator
        assert validate_structural_frequency(1.0) == 1.0
        assert validate_structural_frequency(100.5) == 100.5

    def test_integer_frequencies(self) -> None:
        """Test that integer frequencies are accepted and converted to float."""
        result = validate_structural_frequency(5)
        assert result == 5.0
        assert isinstance(result, float)

    def test_negative_frequency(self) -> None:
        """Test that negative frequencies are rejected."""
        with pytest.raises(ValueError, match="must be non-negative"):
            validate_structural_frequency(-0.1)

        with pytest.raises(ValueError, match="must be non-negative"):
            validate_structural_frequency(-10.0)

    def test_nan_frequency(self) -> None:
        """Test that NaN frequencies are rejected."""
        with pytest.raises(ValueError, match="cannot be NaN"):
            validate_structural_frequency(float("nan"))

    def test_infinite_frequency(self) -> None:
        """Test that infinite frequencies are rejected."""
        with pytest.raises(ValueError, match="cannot be infinite"):
            validate_structural_frequency(float("inf"))

        with pytest.raises(ValueError, match="cannot be infinite"):
            validate_structural_frequency(float("-inf"))

    def test_non_numeric_frequency(self) -> None:
        """Test that non-numeric frequencies are rejected."""
        with pytest.raises(ValueError, match="must be numeric"):
            validate_structural_frequency("0.5")  # type: ignore

        with pytest.raises(ValueError, match="must be numeric"):
            validate_structural_frequency(None)  # type: ignore


class TestValidatePhaseValue:
    """Tests for phase (φ) validation."""

    def test_valid_phases(self) -> None:
        """Test that valid phase values pass validation."""
        assert validate_phase_value(0.0) == 0.0
        assert validate_phase_value(1.57) == 1.57  # π/2
        assert validate_phase_value(3.14) == 3.14  # π

        # Phase at 2π wraps to 0
        two_pi = 2 * math.pi
        result = validate_phase_value(two_pi)
        assert result < 1e-10  # Should be very close to 0

    def test_phase_wrapping(self) -> None:
        """Test that phase wrapping works correctly."""
        two_pi = 2 * math.pi

        # Phase > 2π should wrap
        result = validate_phase_value(two_pi + 1.0)
        assert 0.0 <= result <= two_pi
        assert abs(result - 1.0) < 1e-10

        # Negative phase should wrap to positive
        result = validate_phase_value(-1.0)
        assert 0.0 <= result <= two_pi
        assert abs(result - (two_pi - 1.0)) < 1e-10

    def test_phase_no_wrapping(self) -> None:
        """Test phase validation without wrapping."""
        # Valid phase within range
        assert validate_phase_value(1.57, allow_wrap=False) == 1.57

        # Phase outside range should raise error
        with pytest.raises(ValueError, match="must be in range"):
            validate_phase_value(7.0, allow_wrap=False)

        with pytest.raises(ValueError, match="must be in range"):
            validate_phase_value(-1.0, allow_wrap=False)

    def test_nan_phase(self) -> None:
        """Test that NaN phases are rejected."""
        with pytest.raises(ValueError, match="cannot be NaN"):
            validate_phase_value(float("nan"))

    def test_infinite_phase(self) -> None:
        """Test that infinite phases are rejected."""
        with pytest.raises(ValueError, match="cannot be infinite"):
            validate_phase_value(float("inf"))

    def test_non_numeric_phase(self) -> None:
        """Test that non-numeric phases are rejected."""
        with pytest.raises(ValueError, match="must be numeric"):
            validate_phase_value("1.57")  # type: ignore


class TestValidateCoherenceValue:
    """Tests for coherence C(t) validation."""

    def test_valid_coherence(self) -> None:
        """Test that valid coherence values pass validation."""
        assert validate_coherence_value(0.0) == 0.0  # Minimum
        assert validate_coherence_value(0.5) == 0.5
        assert validate_coherence_value(1.0) == 1.0
        assert validate_coherence_value(10.0) == 10.0  # High coherence

    def test_negative_coherence(self) -> None:
        """Test that negative coherence is rejected."""
        with pytest.raises(ValueError, match="must be non-negative"):
            validate_coherence_value(-0.1)

    def test_nan_coherence(self) -> None:
        """Test that NaN coherence is rejected."""
        with pytest.raises(ValueError, match="cannot be NaN"):
            validate_coherence_value(float("nan"))

    def test_infinite_coherence(self) -> None:
        """Test that infinite coherence is rejected."""
        with pytest.raises(ValueError, match="cannot be infinite"):
            validate_coherence_value(float("inf"))

    def test_non_numeric_coherence(self) -> None:
        """Test that non-numeric coherence is rejected."""
        with pytest.raises(ValueError, match="must be numeric"):
            validate_coherence_value("0.5")  # type: ignore


class TestValidateSenseIndex:
    """Tests for sense index (Si) validation."""

    def test_valid_sense_index(self) -> None:
        """Test that valid sense index values pass validation."""
        assert validate_sense_index(0.0) == 0.0
        assert validate_sense_index(0.5) == 0.5
        assert validate_sense_index(1.0) == 1.0
        assert validate_sense_index(1.5) == 1.5  # Can exceed 1.0

    def test_negative_sense_index(self) -> None:
        """Test that negative sense index is rejected."""
        with pytest.raises(ValueError, match="must be non-negative"):
            validate_sense_index(-0.1)

    def test_nan_sense_index(self) -> None:
        """Test that NaN sense index is rejected."""
        with pytest.raises(ValueError, match="cannot be NaN"):
            validate_sense_index(float("nan"))

    def test_infinite_sense_index(self) -> None:
        """Test that infinite sense index is rejected."""
        with pytest.raises(ValueError, match="cannot be infinite"):
            validate_sense_index(float("inf"))

    def test_non_numeric_sense_index(self) -> None:
        """Test that non-numeric sense index is rejected."""
        with pytest.raises(ValueError, match="must be numeric"):
            validate_sense_index("0.5")  # type: ignore


class TestValidateNodalInput:
    """Tests for complete nodal data validation."""

    def test_valid_nodal_data(self) -> None:
        """Test that valid nodal data passes validation."""
        data = {
            "nu_f": 0.5,
            "phase": 1.57,
            "coherence": 0.8,
            "si": 0.7,
        }
        validated = validate_nodal_input(data)

        assert validated["nu_f"] == 0.5
        assert validated["phase"] == 1.57
        assert validated["coherence"] == 0.8
        assert validated["si"] == 0.7

    def test_partial_nodal_data(self) -> None:
        """Test that partial nodal data is validated correctly."""
        # Only nu_f and phase
        data = {"nu_f": 0.5, "phase": 1.57}
        validated = validate_nodal_input(data)
        assert "nu_f" in validated
        assert "phase" in validated
        assert "coherence" not in validated

    def test_additional_fields_passed_through(self) -> None:
        """Test that additional fields are passed through unchanged."""
        data = {
            "nu_f": 0.5,
            "node_id": "node_123",
            "epi": [1.0, 2.0, 3.0],
            "custom_field": "value",
        }
        validated = validate_nodal_input(data)

        assert validated["nu_f"] == 0.5
        assert validated["node_id"] == "node_123"
        assert validated["epi"] == [1.0, 2.0, 3.0]
        assert validated["custom_field"] == "value"

    def test_sense_index_alternative_key(self) -> None:
        """Test that both 'si' and 'sense_index' keys are recognized."""
        # Test with 'si'
        data1 = {"si": 0.7}
        validated1 = validate_nodal_input(data1)
        assert validated1["si"] == 0.7

        # Test with 'sense_index'
        data2 = {"sense_index": 0.7}
        validated2 = validate_nodal_input(data2)
        assert validated2["sense_index"] == 0.7

    def test_invalid_nodal_data(self) -> None:
        """Test that invalid nodal data is rejected."""
        # Invalid nu_f
        with pytest.raises(ValueError, match="must be non-negative"):
            validate_nodal_input({"nu_f": -0.5})

        # Invalid phase
        with pytest.raises(ValueError, match="cannot be NaN"):
            validate_nodal_input({"phase": float("nan")})

        # Invalid coherence
        with pytest.raises(ValueError, match="must be non-negative"):
            validate_nodal_input({"coherence": -1.0})

        # Invalid sense index
        with pytest.raises(ValueError, match="cannot be infinite"):
            validate_nodal_input({"si": float("inf")})

    def test_empty_data(self) -> None:
        """Test that empty data passes validation (no fields to validate)."""
        validated = validate_nodal_input({})
        assert validated == {}

    def test_phase_wrapping_in_nodal_data(self) -> None:
        """Test that phase wrapping occurs in nodal validation."""
        data = {"phase": 7.0}  # Greater than 2π
        validated = validate_nodal_input(data)

        # Should be wrapped to [0, 2π]
        two_pi = 2 * math.pi
        assert 0.0 <= validated["phase"] <= two_pi


class TestValidationIntegration:
    """Integration tests for validation utilities."""

    def test_complete_nfr_node_validation(self) -> None:
        """Test validation of a complete NFR node data structure."""
        node_data = {
            "node_id": "nfr_001",
            "nu_f": 0.75,  # Structural frequency
            "phase": 1.57,  # π/2
            "coherence": 0.85,
            "si": 0.9,
            "delta_nfr": 0.05,
            "epi": [1.5, 2.3, 0.8],
            "neighbors": ["nfr_002", "nfr_003"],
        }

        validated = validate_nodal_input(node_data)

        # Check validated fields
        assert validated["nu_f"] == 0.75
        assert validated["phase"] == 1.57
        assert validated["coherence"] == 0.85
        assert validated["si"] == 0.9

        # Check pass-through fields
        assert validated["node_id"] == "nfr_001"
        assert validated["delta_nfr"] == 0.05
        assert validated["epi"] == [1.5, 2.3, 0.8]
        assert validated["neighbors"] == ["nfr_002", "nfr_003"]

    def test_boundary_values(self) -> None:
        """Test validation at boundary values."""
        # Zero values (valid)
        assert validate_structural_frequency(0.0) == 0.0
        assert validate_phase_value(0.0) == 0.0
        assert validate_coherence_value(0.0) == 0.0
        assert validate_sense_index(0.0) == 0.0

        # Phase at 2π wraps to 0 (since it's at the boundary)
        two_pi = 2 * math.pi
        result = validate_phase_value(two_pi)
        assert result < 1e-10  # Should wrap to ~0

        # Very large values (valid for most metrics)
        assert validate_structural_frequency(1000.0) == 1000.0
        assert validate_coherence_value(100.0) == 100.0
        assert validate_sense_index(10.0) == 10.0

    def test_tnfr_operator_states(self) -> None:
        """Test validation for different TNFR operator states."""
        # Silence operator: νf ≈ 0
        silence_data = {"nu_f": 0.0, "phase": 0.0}
        validated = validate_nodal_input(silence_data)
        assert validated["nu_f"] == 0.0

        # High resonance: high νf and coherence
        resonance_data = {"nu_f": 0.95, "coherence": 0.98}
        validated = validate_nodal_input(resonance_data)
        assert validated["nu_f"] == 0.95
        assert validated["coherence"] == 0.98

        # Phase transition: phase change
        transition_data = {"phase": 3.14}  # π
        validated = validate_nodal_input(transition_data)
        assert abs(validated["phase"] - 3.14) < 1e-10
