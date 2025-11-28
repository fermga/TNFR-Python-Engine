"""Tests for contextual error messages.

This module tests the enhanced error handling system to ensure:
1. Error messages are clear and helpful
2. Suggestions are accurate
3. Fuzzy matching works correctly
4. TNFR invariants are referenced correctly
5. Documentation links are valid
"""

import pytest
from tnfr.errors import (
    TNFRUserError,
    OperatorSequenceError,
    NetworkConfigError,
    PhaseError,
    CoherenceError,
    FrequencyError,
)


class TestTNFRUserError:
    """Test base error class functionality."""

    def test_basic_error(self):
        """Test basic error with just a message."""
        with pytest.raises(TNFRUserError) as exc_info:
            raise TNFRUserError("Something went wrong")

        assert "Something went wrong" in str(exc_info.value)

    def test_error_with_suggestion(self):
        """Test error with suggestion included."""
        with pytest.raises(TNFRUserError) as exc_info:
            raise TNFRUserError("Invalid parameter", suggestion="Use parameter X instead")

        error_msg = str(exc_info.value)
        assert "Invalid parameter" in error_msg
        assert "Suggestion:" in error_msg
        assert "Use parameter X instead" in error_msg

    def test_error_with_docs_url(self):
        """Test error with documentation link."""
        with pytest.raises(TNFRUserError) as exc_info:
            raise TNFRUserError("Configuration error", docs_url="https://example.com/docs")

        error_msg = str(exc_info.value)
        assert "Configuration error" in error_msg
        assert "Documentation:" in error_msg
        assert "https://example.com/docs" in error_msg

    def test_error_with_context(self):
        """Test error with additional context."""
        with pytest.raises(TNFRUserError) as exc_info:
            raise TNFRUserError("Operation failed", context={"node": "n1", "value": 42})

        error_msg = str(exc_info.value)
        assert "Operation failed" in error_msg
        assert "Context:" in error_msg
        assert "node: n1" in error_msg
        assert "value: 42" in error_msg


class TestOperatorSequenceError:
    """Test operator sequence validation errors."""

    def test_invalid_operator_basic(self):
        """Test error for invalid operator name."""
        with pytest.raises(OperatorSequenceError) as exc_info:
            raise OperatorSequenceError("invalid_op")

        error_msg = str(exc_info.value)
        assert "invalid_op" in error_msg
        assert "cannot be applied" in error_msg

    def test_fuzzy_matching_typo(self):
        """Test fuzzy matching suggests correct operator."""
        with pytest.raises(OperatorSequenceError) as exc_info:
            raise OperatorSequenceError("emision")  # Missing 's'

        error_msg = str(exc_info.value)
        assert "emision" in error_msg
        assert "emission" in error_msg.lower()  # Should suggest correction

    def test_fuzzy_matching_alias(self):
        """Test fuzzy matching works with aliases."""
        with pytest.raises(OperatorSequenceError) as exc_info:
            raise OperatorSequenceError("emitt")  # Close to 'emit'

        error_msg = str(exc_info.value)
        # Should suggest 'emit' or 'emission'
        assert "emit" in error_msg.lower() or "emission" in error_msg.lower()

    def test_sequence_context(self):
        """Test error includes sequence context."""
        with pytest.raises(OperatorSequenceError) as exc_info:
            raise OperatorSequenceError(
                "bad_op", sequence_so_far=["emission", "reception", "coherence"]
            )

        error_msg = str(exc_info.value)
        assert "bad_op" in error_msg
        assert "emission" in error_msg
        assert "reception" in error_msg
        assert "coherence" in error_msg

    def test_valid_next_operators(self):
        """Test error suggests valid next operators."""
        with pytest.raises(OperatorSequenceError) as exc_info:
            raise OperatorSequenceError("invalid", valid_next=["resonance", "silence"])

        error_msg = str(exc_info.value)
        assert "Valid next operators" in error_msg
        assert "resonance" in error_msg
        assert "silence" in error_msg

    def test_valid_operators_listed(self):
        """Test that valid operators are mentioned."""
        with pytest.raises(OperatorSequenceError) as exc_info:
            raise OperatorSequenceError("xyz")

        error_msg = str(exc_info.value)
        # Should mention canonical operators
        assert "emission" in error_msg or "13 canonical" in error_msg


class TestNetworkConfigError:
    """Test network configuration validation errors."""

    def test_invalid_vf_negative(self):
        """Test error for negative structural frequency."""
        with pytest.raises(NetworkConfigError) as exc_info:
            raise NetworkConfigError("vf", -0.5)

        error_msg = str(exc_info.value)
        assert "vf" in error_msg
        assert "-0.5" in error_msg
        assert "Hz_str" in error_msg

    def test_invalid_vf_with_range(self):
        """Test error includes valid range."""
        with pytest.raises(NetworkConfigError) as exc_info:
            raise NetworkConfigError("vf", 150, (0.01, 100.0))

        error_msg = str(exc_info.value)
        assert "vf" in error_msg
        assert "0.01" in error_msg
        assert "100.0" in error_msg

    def test_predefined_constraints(self):
        """Test that predefined constraints are used."""
        with pytest.raises(NetworkConfigError) as exc_info:
            raise NetworkConfigError("phase", -1.0)

        error_msg = str(exc_info.value)
        assert "phase" in error_msg
        assert "radians" in error_msg

    def test_edge_probability_constraint(self):
        """Test edge probability validation."""
        with pytest.raises(NetworkConfigError) as exc_info:
            raise NetworkConfigError("edge_probability", 1.5)

        error_msg = str(exc_info.value)
        assert "edge_probability" in error_msg
        assert "0.0" in error_msg
        assert "1.0" in error_msg

    def test_custom_parameter(self):
        """Test error for custom parameter not in predefined list."""
        with pytest.raises(NetworkConfigError) as exc_info:
            raise NetworkConfigError(
                "custom_param", 999, valid_range=(0, 100), reason="Must be in valid range"
            )

        error_msg = str(exc_info.value)
        assert "custom_param" in error_msg
        assert "999" in error_msg
        assert "0" in error_msg
        assert "100" in error_msg
        assert "Must be in valid range" in error_msg


class TestPhaseError:
    """Test phase synchrony validation errors."""

    def test_phase_incompatibility(self):
        """Test error for phase mismatch between nodes."""
        with pytest.raises(PhaseError) as exc_info:
            raise PhaseError("n1", "n2", 0.5, 2.8, 0.5)

        error_msg = str(exc_info.value)
        assert "n1" in error_msg
        assert "n2" in error_msg
        assert "phase" in error_msg.lower()

    def test_phase_values_included(self):
        """Test that phase values are shown in error."""
        with pytest.raises(PhaseError) as exc_info:
            raise PhaseError("node_a", "node_b", 1.2, 3.0, 0.5)

        error_msg = str(exc_info.value)
        assert "1.2" in error_msg or "1.200" in error_msg
        assert "3.0" in error_msg or "3.000" in error_msg

    def test_phase_difference_calculated(self):
        """Test that phase difference is calculated and shown."""
        with pytest.raises(PhaseError) as exc_info:
            raise PhaseError("n1", "n2", 1.0, 3.0, 0.5)

        error_msg = str(exc_info.value)
        # Phase difference should be ~2.0
        assert "difference" in error_msg.lower()

    def test_threshold_included(self):
        """Test that threshold is mentioned in error."""
        with pytest.raises(PhaseError) as exc_info:
            raise PhaseError("n1", "n2", 0.0, 1.0, 0.3)

        error_msg = str(exc_info.value)
        assert "0.3" in error_msg or "0.300" in error_msg
        assert "threshold" in error_msg.lower()


class TestCoherenceError:
    """Test coherence monotonicity validation errors."""

    def test_coherence_decrease(self):
        """Test error when coherence unexpectedly decreases."""
        with pytest.raises(CoherenceError) as exc_info:
            raise CoherenceError("coherence", 0.85, 0.42)

        error_msg = str(exc_info.value)
        assert "coherence" in error_msg.lower()
        assert "0.85" in error_msg
        assert "0.42" in error_msg

    def test_coherence_decrease_percentage(self):
        """Test that percentage decrease is calculated."""
        with pytest.raises(CoherenceError) as exc_info:
            raise CoherenceError("test_operation", 1.0, 0.5)

        error_msg = str(exc_info.value)
        # Should show ~50% decrease
        assert "%" in error_msg
        assert "decrease" in error_msg.lower()

    def test_operation_name_included(self):
        """Test that operation name is included."""
        with pytest.raises(CoherenceError) as exc_info:
            raise CoherenceError("my_operation", 0.8, 0.6)

        error_msg = str(exc_info.value)
        assert "my_operation" in error_msg

    def test_node_specific_error(self):
        """Test error for node-specific coherence issue."""
        with pytest.raises(CoherenceError) as exc_info:
            raise CoherenceError("operation", 0.9, 0.3, node_id="node_123")

        error_msg = str(exc_info.value)
        assert "node_123" in error_msg

    def test_invariant_mentioned(self):
        """Test that TNFR invariant is referenced."""
        with pytest.raises(CoherenceError) as exc_info:
            raise CoherenceError("op", 0.7, 0.4)

        error_msg = str(exc_info.value)
        assert "invariant" in error_msg.lower()


class TestFrequencyError:
    """Test structural frequency validation errors."""

    def test_negative_frequency(self):
        """Test error for negative structural frequency."""
        with pytest.raises(FrequencyError) as exc_info:
            raise FrequencyError("n1", -0.5)

        error_msg = str(exc_info.value)
        assert "n1" in error_msg
        assert "-0.5" in error_msg or "negative" in error_msg.lower()
        assert "positive" in error_msg.lower()

    def test_zero_frequency(self):
        """Test error for zero structural frequency."""
        with pytest.raises(FrequencyError) as exc_info:
            raise FrequencyError("node_x", 0.0)

        error_msg = str(exc_info.value)
        assert "node_x" in error_msg
        assert "positive" in error_msg.lower()

    def test_very_high_frequency(self):
        """Test warning for very high frequency."""
        with pytest.raises(FrequencyError) as exc_info:
            raise FrequencyError("n1", 150.0)

        error_msg = str(exc_info.value)
        assert "150" in error_msg
        assert "high" in error_msg.lower()

    def test_hz_str_units(self):
        """Test that Hz_str units are mentioned."""
        with pytest.raises(FrequencyError) as exc_info:
            raise FrequencyError("n1", -1.0)

        error_msg = str(exc_info.value)
        assert "Hz_str" in error_msg

    def test_operation_context(self):
        """Test error includes operation context."""
        with pytest.raises(FrequencyError) as exc_info:
            raise FrequencyError("n1", -0.5, operation="emission")

        error_msg = str(exc_info.value)
        assert "emission" in error_msg

    def test_typical_range_suggested(self):
        """Test that typical range is suggested."""
        with pytest.raises(FrequencyError) as exc_info:
            raise FrequencyError("n1", 0.0)

        error_msg = str(exc_info.value)
        assert "0.1" in error_msg or "range" in error_msg.lower()


class TestErrorIntegration:
    """Test integration and edge cases."""

    def test_all_errors_inherit_from_base(self):
        """Test that all errors inherit from TNFRUserError."""
        assert issubclass(OperatorSequenceError, TNFRUserError)
        assert issubclass(NetworkConfigError, TNFRUserError)
        assert issubclass(PhaseError, TNFRUserError)
        assert issubclass(CoherenceError, TNFRUserError)
        assert issubclass(FrequencyError, TNFRUserError)

    def test_all_errors_inherit_from_exception(self):
        """Test that all errors inherit from base Exception."""
        assert issubclass(TNFRUserError, Exception)

    def test_error_can_be_caught_as_base_error(self):
        """Test that specific errors can be caught as TNFRUserError."""
        with pytest.raises(TNFRUserError):
            raise OperatorSequenceError("bad_op")

        with pytest.raises(TNFRUserError):
            raise NetworkConfigError("param", 999)

        with pytest.raises(TNFRUserError):
            raise PhaseError("n1", "n2", 0, 1, 0.5)

    def test_multiple_context_items(self):
        """Test error with multiple context items."""
        with pytest.raises(TNFRUserError) as exc_info:
            raise TNFRUserError(
                "Complex error",
                context={
                    "parameter1": "value1",
                    "parameter2": 42,
                    "parameter3": [1, 2, 3],
                },
            )

        error_msg = str(exc_info.value)
        assert "parameter1" in error_msg
        assert "parameter2" in error_msg
        assert "parameter3" in error_msg

    def test_empty_context(self):
        """Test error with empty context dictionary."""
        with pytest.raises(TNFRUserError) as exc_info:
            raise TNFRUserError("Error", context={})

        error_msg = str(exc_info.value)
        assert "Error" in error_msg
        # Should not show empty context section

    def test_none_optional_parameters(self):
        """Test error with None for optional parameters."""
        with pytest.raises(TNFRUserError) as exc_info:
            raise TNFRUserError("Error", suggestion=None, docs_url=None, context=None)

        error_msg = str(exc_info.value)
        assert "Error" in error_msg
        # Should handle None gracefully
