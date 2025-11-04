"""Tests for input validation of TNFR structural operators.

These tests verify that input validation functions properly enforce bounds,
type safety, and security constraints for structural operator parameters.
"""

import math

import networkx as nx
import pytest

from tnfr.types import Glyph
from tnfr.validation.input_validation import (
    ValidationError,
    validate_dnfr_value,
    validate_epi_value,
    validate_glyph,
    validate_glyph_factors,
    validate_node_id,
    validate_operator_parameters,
    validate_theta_value,
    validate_tnfr_graph,
    validate_vf_value,
)


class TestValidateEPIValue:
    """Test EPI (Primary Information Structure) validation."""

    def test_valid_epi_float(self):
        """Test valid float EPI values."""
        assert validate_epi_value(0.5) == 0.5
        assert validate_epi_value(0.0) == 0.0
        assert validate_epi_value(1.0) == 1.0

    def test_valid_epi_complex(self):
        """Test valid complex EPI values."""
        result = validate_epi_value(0.5 + 0.3j)
        assert result == 0.5 + 0.3j

    def test_epi_complex_not_allowed(self):
        """Test complex EPI rejected when not allowed."""
        with pytest.raises(ValidationError, match="must be real-valued"):
            validate_epi_value(0.5 + 0.3j, allow_complex=False)

    def test_epi_non_numeric(self):
        """Test non-numeric EPI rejected."""
        with pytest.raises(ValidationError, match="must be numeric"):
            validate_epi_value("0.5")

        with pytest.raises(ValidationError, match="must be numeric"):
            validate_epi_value([0.5])

    def test_epi_nan_rejected(self):
        """Test NaN EPI rejected."""
        with pytest.raises(ValidationError, match="cannot be nan or inf"):
            validate_epi_value(float("nan"))

    def test_epi_inf_rejected(self):
        """Test infinite EPI rejected."""
        with pytest.raises(ValidationError, match="cannot be nan or inf"):
            validate_epi_value(float("inf"))

        with pytest.raises(ValidationError, match="cannot be nan or inf"):
            validate_epi_value(float("-inf"))

    def test_epi_below_minimum(self):
        """Test EPI below minimum bound rejected."""
        config = {"EPI_MIN": 0.1, "EPI_MAX": 1.0}
        with pytest.raises(ValidationError, match="below minimum bound"):
            validate_epi_value(0.05, config=config)

    def test_epi_above_maximum(self):
        """Test EPI above maximum bound rejected."""
        config = {"EPI_MIN": 0.0, "EPI_MAX": 1.0}
        with pytest.raises(ValidationError, match="exceeds maximum bound"):
            validate_epi_value(1.5, config=config)

    def test_epi_custom_bounds(self):
        """Test EPI validation with custom bounds."""
        config = {"EPI_MIN": 0.2, "EPI_MAX": 0.8}
        assert validate_epi_value(0.5, config=config) == 0.5

        with pytest.raises(ValidationError):
            validate_epi_value(0.1, config=config)

        with pytest.raises(ValidationError):
            validate_epi_value(0.9, config=config)

    def test_epi_negative_magnitude(self):
        """Test negative EPI values checked by magnitude."""
        config = {"EPI_MIN": 0.1, "EPI_MAX": 1.0}
        assert validate_epi_value(-0.5, config=config) == -0.5

        with pytest.raises(ValidationError, match="below minimum bound"):
            validate_epi_value(-0.05, config=config)


class TestValidateVFValue:
    """Test νf (structural frequency) validation."""

    def test_valid_vf(self):
        """Test valid νf values."""
        assert validate_vf_value(1.0) == 1.0
        assert validate_vf_value(0.0) == 0.0
        # Use a value within default VF_MAX bound
        assert validate_vf_value(0.5) == 0.5

    def test_vf_non_numeric(self):
        """Test non-numeric νf rejected."""
        with pytest.raises(ValidationError, match="must be numeric"):
            validate_vf_value("1.0")

    def test_vf_complex_rejected(self):
        """Test complex νf rejected."""
        with pytest.raises(ValidationError, match="must be numeric"):
            validate_vf_value(1.0 + 0.5j)

    def test_vf_nan_rejected(self):
        """Test NaN νf rejected."""
        with pytest.raises(ValidationError, match="cannot be nan or inf"):
            validate_vf_value(float("nan"))

    def test_vf_inf_rejected(self):
        """Test infinite νf rejected."""
        with pytest.raises(ValidationError, match="cannot be nan or inf"):
            validate_vf_value(float("inf"))

    def test_vf_negative_rejected(self):
        """Test negative νf rejected."""
        with pytest.raises(ValidationError, match="must be non-negative"):
            validate_vf_value(-0.5)

    def test_vf_below_minimum(self):
        """Test νf below minimum bound rejected."""
        config = {"VF_MIN": 0.5, "VF_MAX": 10.0}
        with pytest.raises(ValidationError, match="below minimum bound"):
            validate_vf_value(0.3, config=config)

    def test_vf_above_maximum(self):
        """Test νf above maximum bound rejected."""
        config = {"VF_MIN": 0.0, "VF_MAX": 5.0}
        with pytest.raises(ValidationError, match="exceeds maximum bound"):
            validate_vf_value(6.0, config=config)

    def test_vf_custom_bounds(self):
        """Test νf validation with custom bounds."""
        config = {"VF_MIN": 1.0, "VF_MAX": 3.0}
        assert validate_vf_value(2.0, config=config) == 2.0

        with pytest.raises(ValidationError):
            validate_vf_value(0.5, config=config)

        with pytest.raises(ValidationError):
            validate_vf_value(4.0, config=config)


class TestValidateThetaValue:
    """Test θ (phase) validation."""

    def test_valid_theta(self):
        """Test valid θ values."""
        assert validate_theta_value(0.0) == 0.0
        assert abs(validate_theta_value(math.pi / 2) - math.pi / 2) < 1e-10

    def test_theta_normalization(self):
        """Test θ normalization to [-π, π]."""
        result = validate_theta_value(3 * math.pi, normalize=True)
        # 3π normalizes to -π (wraps around)
        assert abs(result - (-math.pi)) < 1e-10

        result = validate_theta_value(-3 * math.pi, normalize=True)
        # -3π normalizes to -π (wraps around)
        assert abs(result - (-math.pi)) < 1e-10

    def test_theta_no_normalization(self):
        """Test θ without normalization."""
        value = 3 * math.pi
        result = validate_theta_value(value, normalize=False)
        assert result == value

    def test_theta_non_numeric(self):
        """Test non-numeric θ rejected."""
        with pytest.raises(ValidationError, match="must be numeric"):
            validate_theta_value("0.5")

    def test_theta_complex_rejected(self):
        """Test complex θ rejected."""
        with pytest.raises(ValidationError, match="must be numeric"):
            validate_theta_value(1.0 + 0.5j)

    def test_theta_nan_rejected(self):
        """Test NaN θ rejected."""
        with pytest.raises(ValidationError, match="cannot be nan or inf"):
            validate_theta_value(float("nan"))

    def test_theta_inf_rejected(self):
        """Test infinite θ rejected."""
        with pytest.raises(ValidationError, match="cannot be nan or inf"):
            validate_theta_value(float("inf"))


class TestValidateDNFRValue:
    """Test ΔNFR (reorganization operator) validation."""

    def test_valid_dnfr(self):
        """Test valid ΔNFR values."""
        assert validate_dnfr_value(0.1) == 0.1
        assert validate_dnfr_value(0.0) == 0.0
        assert validate_dnfr_value(-0.5) == -0.5

    def test_dnfr_non_numeric(self):
        """Test non-numeric ΔNFR rejected."""
        with pytest.raises(ValidationError, match="must be numeric"):
            validate_dnfr_value("0.1")

    def test_dnfr_complex_rejected(self):
        """Test complex ΔNFR rejected."""
        with pytest.raises(ValidationError, match="must be numeric"):
            validate_dnfr_value(0.1 + 0.2j)

    def test_dnfr_nan_rejected(self):
        """Test NaN ΔNFR rejected."""
        with pytest.raises(ValidationError, match="cannot be nan or inf"):
            validate_dnfr_value(float("nan"))

    def test_dnfr_inf_rejected(self):
        """Test infinite ΔNFR rejected."""
        with pytest.raises(ValidationError, match="cannot be nan or inf"):
            validate_dnfr_value(float("inf"))

    def test_dnfr_exceeds_maximum(self):
        """Test ΔNFR exceeding maximum bound rejected."""
        config = {"DNFR_MAX": 1.0}
        with pytest.raises(ValidationError, match="exceeds maximum bound"):
            validate_dnfr_value(1.5, config=config)

        with pytest.raises(ValidationError, match="exceeds maximum bound"):
            validate_dnfr_value(-1.5, config=config)

    def test_dnfr_custom_bounds(self):
        """Test ΔNFR validation with custom bounds."""
        config = {"DNFR_MAX": 0.5}
        assert validate_dnfr_value(0.3, config=config) == 0.3
        assert validate_dnfr_value(-0.3, config=config) == -0.3

        with pytest.raises(ValidationError):
            validate_dnfr_value(0.6, config=config)


class TestValidateNodeId:
    """Test NodeId validation."""

    def test_valid_string_node_id(self):
        """Test valid string NodeId."""
        assert validate_node_id("node_1") == "node_1"
        assert validate_node_id("test-node") == "test-node"
        assert validate_node_id("Node123") == "Node123"

    def test_valid_numeric_node_id(self):
        """Test valid numeric NodeId."""
        assert validate_node_id(42) == 42
        assert validate_node_id(0) == 0

    def test_valid_tuple_node_id(self):
        """Test valid tuple NodeId."""
        assert validate_node_id((1, 2)) == (1, 2)

    def test_unhashable_node_id_rejected(self):
        """Test unhashable NodeId rejected."""
        with pytest.raises(ValidationError, match="must be hashable"):
            validate_node_id([1, 2, 3])

        with pytest.raises(ValidationError, match="must be hashable"):
            validate_node_id({"key": "value"})

    def test_node_id_control_characters_rejected(self):
        """Test NodeId with control characters rejected."""
        with pytest.raises(ValidationError, match="cannot contain control characters"):
            validate_node_id("node\x00id")

        with pytest.raises(ValidationError, match="cannot contain control characters"):
            validate_node_id("node\x1fid")

    def test_node_id_injection_patterns_rejected(self):
        """Test NodeId with injection patterns rejected."""
        with pytest.raises(ValidationError, match="suspicious pattern"):
            validate_node_id("<script>alert('xss')</script>")

        with pytest.raises(ValidationError, match="suspicious pattern"):
            validate_node_id("javascript:alert(1)")

        with pytest.raises(ValidationError, match="suspicious pattern"):
            validate_node_id("onclick=alert(1)")

        with pytest.raises(ValidationError, match="suspicious pattern"):
            validate_node_id("${injection}")

        with pytest.raises(ValidationError, match="suspicious pattern"):
            validate_node_id("node`id")


class TestValidateGlyph:
    """Test Glyph enumeration validation."""

    def test_valid_glyph_enum(self):
        """Test valid Glyph enumeration."""
        assert validate_glyph(Glyph.AL) == Glyph.AL
        assert validate_glyph(Glyph.EN) == Glyph.EN
        assert validate_glyph(Glyph.IL) == Glyph.IL

    def test_valid_glyph_string(self):
        """Test valid Glyph string conversion."""
        assert validate_glyph("AL") == Glyph.AL
        assert validate_glyph("EN") == Glyph.EN
        assert validate_glyph("IL") == Glyph.IL

    def test_invalid_glyph_rejected(self):
        """Test invalid Glyph rejected."""
        with pytest.raises(ValidationError, match="Invalid glyph value"):
            validate_glyph("INVALID")

        with pytest.raises(ValidationError, match="Invalid glyph value"):
            validate_glyph(123)


class TestValidateTNFRGraph:
    """Test TNFRGraph validation."""

    def test_valid_graph(self):
        """Test valid TNFRGraph."""
        G = nx.Graph()
        assert validate_tnfr_graph(G) is G

    def test_valid_digraph(self):
        """Test valid DiGraph."""
        G = nx.DiGraph()
        assert validate_tnfr_graph(G) is G

    def test_invalid_graph_type_rejected(self):
        """Test non-graph types rejected."""
        with pytest.raises(ValidationError, match="Expected TNFRGraph"):
            validate_tnfr_graph("not a graph")

        with pytest.raises(ValidationError, match="Expected TNFRGraph"):
            validate_tnfr_graph([])

    def test_graph_without_graph_attribute(self):
        """Test graph without 'graph' attribute rejected."""

        class FakeGraph:
            pass

        with pytest.raises(ValidationError, match="Expected TNFRGraph"):
            validate_tnfr_graph(FakeGraph())


class TestValidateGlyphFactors:
    """Test glyph factors validation."""

    def test_valid_glyph_factors(self):
        """Test valid glyph factors."""
        factors = {"AL_boost": 0.1, "EN_mix": 0.25}
        result = validate_glyph_factors(factors)
        assert result == {"AL_boost": 0.1, "EN_mix": 0.25}

    def test_empty_glyph_factors(self):
        """Test empty glyph factors."""
        result = validate_glyph_factors({})
        assert result == {}

    def test_glyph_factors_non_mapping_rejected(self):
        """Test non-mapping glyph factors rejected."""
        with pytest.raises(ValidationError, match="must be a mapping"):
            validate_glyph_factors([("AL_boost", 0.1)])

    def test_glyph_factors_non_string_key_rejected(self):
        """Test non-string key rejected."""
        with pytest.raises(ValidationError, match="key must be string"):
            validate_glyph_factors({123: 0.1})

    def test_glyph_factors_non_numeric_value_rejected(self):
        """Test non-numeric value rejected."""
        with pytest.raises(ValidationError, match="value must be numeric"):
            validate_glyph_factors({"AL_boost": "0.1"})

    def test_glyph_factors_nan_value_rejected(self):
        """Test NaN value rejected."""
        with pytest.raises(ValidationError, match="cannot be nan or inf"):
            validate_glyph_factors({"AL_boost": float("nan")})

    def test_glyph_factors_inf_value_rejected(self):
        """Test infinite value rejected."""
        with pytest.raises(ValidationError, match="cannot be nan or inf"):
            validate_glyph_factors({"AL_boost": float("inf")})

    def test_glyph_factors_required_keys(self):
        """Test required keys validation."""
        factors = {"AL_boost": 0.1}
        required = {"AL_boost", "EN_mix"}

        with pytest.raises(ValidationError, match="Missing required glyph factor keys"):
            validate_glyph_factors(factors, required_keys=required)

    def test_glyph_factors_required_keys_satisfied(self):
        """Test required keys satisfied."""
        factors = {"AL_boost": 0.1, "EN_mix": 0.25}
        required = {"AL_boost"}
        result = validate_glyph_factors(factors, required_keys=required)
        assert result == factors


class TestValidateOperatorParameters:
    """Test operator parameters validation."""

    def test_validate_epi_parameter(self):
        """Test EPI parameter validation."""
        params = {"epi": 0.5}
        result = validate_operator_parameters(params)
        assert result == {"epi": 0.5}

    def test_validate_vf_parameter(self):
        """Test νf parameter validation."""
        params = {"vf": 1.0}
        result = validate_operator_parameters(params)
        assert result == {"vf": 1.0}

    def test_validate_theta_parameter(self):
        """Test θ parameter validation."""
        params = {"theta": math.pi / 2}
        result = validate_operator_parameters(params)
        assert abs(result["theta"] - math.pi / 2) < 1e-10

    def test_validate_dnfr_parameter(self):
        """Test ΔNFR parameter validation."""
        params = {"dnfr": 0.1}
        result = validate_operator_parameters(params)
        assert result == {"dnfr": 0.1}

    def test_validate_node_parameter(self):
        """Test node parameter validation."""
        params = {"node": "node_1"}
        result = validate_operator_parameters(params)
        assert result == {"node": "node_1"}

    def test_validate_glyph_parameter(self):
        """Test glyph parameter validation."""
        params = {"glyph": Glyph.AL}
        result = validate_operator_parameters(params)
        assert result == {"glyph": Glyph.AL}

    def test_validate_graph_parameter(self):
        """Test graph parameter validation."""
        G = nx.Graph()
        params = {"G": G}
        result = validate_operator_parameters(params)
        assert result["G"] is G

    def test_validate_glyph_factors_parameter(self):
        """Test glyph_factors parameter validation."""
        factors = {"AL_boost": 0.1}
        params = {"glyph_factors": factors}
        result = validate_operator_parameters(params)
        assert result == {"glyph_factors": factors}

    def test_validate_multiple_parameters(self):
        """Test multiple parameters validation."""
        params = {
            "epi": 0.5,
            "vf": 1.0,
            "theta": 0.0,
            "dnfr": 0.1,
            "node": "test",
        }
        result = validate_operator_parameters(params)
        assert result["epi"] == 0.5
        assert result["vf"] == 1.0
        assert result["theta"] == 0.0
        assert result["dnfr"] == 0.1
        assert result["node"] == "test"

    def test_validate_unknown_parameter_passthrough(self):
        """Test unknown parameters passed through unchanged."""
        params = {"custom_param": "value"}
        result = validate_operator_parameters(params)
        assert result == {"custom_param": "value"}

    def test_validate_with_config(self):
        """Test validation with custom configuration."""
        config = {"EPI_MAX": 0.8}
        params = {"epi": 0.7}
        result = validate_operator_parameters(params, config=config)
        assert result == {"epi": 0.7}

        params_invalid = {"epi": 0.9}
        with pytest.raises(ValidationError):
            validate_operator_parameters(params_invalid, config=config)


class TestValidationErrorAttributes:
    """Test ValidationError attributes."""

    def test_validation_error_basic(self):
        """Test basic ValidationError."""
        err = ValidationError("Test error")
        assert str(err) == "Test error"
        assert err.parameter is None
        assert err.value is None
        assert err.constraint is None

    def test_validation_error_with_attributes(self):
        """Test ValidationError with attributes."""
        err = ValidationError(
            "Invalid value",
            parameter="epi",
            value=10.0,
            constraint="<= 1.0",
        )
        assert str(err) == "Invalid value"
        assert err.parameter == "epi"
        assert err.value == 10.0
        assert err.constraint == "<= 1.0"


class TestSecurityPatterns:
    """Test security-specific validation patterns."""

    def test_prevent_xss_in_node_id(self):
        """Test XSS prevention in NodeId."""
        malicious_ids = [
            "<img src=x onerror=alert(1)>",
            "<svg onload=alert(1)>",
            "<iframe src=javascript:alert(1)>",
        ]

        for node_id in malicious_ids:
            with pytest.raises(ValidationError):
                validate_node_id(node_id)

    def test_prevent_template_injection_in_node_id(self):
        """Test template injection prevention."""
        with pytest.raises(ValidationError):
            validate_node_id("${7*7}")

    def test_safe_node_ids_allowed(self):
        """Test that safe NodeIds are not blocked."""
        safe_ids = [
            "node-123",
            "test_node",
            "Node.1",
            "n@123",  # @ is safe
            "node#5",  # # is safe
        ]

        for node_id in safe_ids:
            assert validate_node_id(node_id) == node_id

    def test_numeric_overflow_prevention(self):
        """Test numeric overflow handling."""
        # Very large but finite numbers should be caught by bounds
        config = {"EPI_MAX": 1.0}
        with pytest.raises(ValidationError):
            validate_epi_value(1e100, config=config)

    def test_type_confusion_prevention(self):
        """Test prevention of type confusion attacks."""
        # Ensure strings that look like numbers are rejected
        with pytest.raises(ValidationError):
            validate_vf_value("1.0")

        with pytest.raises(ValidationError):
            validate_epi_value("0.5")

        with pytest.raises(ValidationError):
            validate_dnfr_value("0.1")
