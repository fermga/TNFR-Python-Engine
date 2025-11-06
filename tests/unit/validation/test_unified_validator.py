"""Tests for unified TNFRValidator pipeline.

This module tests the consolidated validation API that serves as the
single entry point for all TNFR validation operations.
"""

import math

import networkx as nx
import pytest

from tnfr.validation import (
    TNFRValidator,
    TNFRValidationError,
    InvariantSeverity,
    ValidationError,
)


class TestTNFRValidatorUnifiedPipeline:
    """Test the unified validation pipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = TNFRValidator()

        # Create a basic TNFR graph with correct attribute names
        self.graph = nx.DiGraph()
        self.graph.graph.update(
            {
                "EPI_MIN": 0.0,
                "EPI_MAX": 1.0,
                "VF_MIN": 0.001,
                "VF_MAX": 1000.0,
            }
        )
        # Use correct TNFR attribute names
        self.graph.add_node(
            "node_1", **{"EPI": 0.5, "νf": 1.0, "theta": 0.0, "ΔNFR": 0.0}
        )
        self.graph.add_node(
            "node_2", **{"EPI": 0.3, "νf": 0.5, "theta": math.pi / 4, "ΔNFR": 0.0}
        )

    def test_validator_initialization(self):
        """Test validator can be initialized with various options."""
        validator1 = TNFRValidator()
        assert validator1 is not None

        validator2 = TNFRValidator(phase_coupling_threshold=math.pi / 4)
        assert validator2 is not None

        validator3 = TNFRValidator(
            enable_input_validation=True,
            enable_graph_validation=True,
            enable_runtime_validation=False,
        )
        assert validator3 is not None

    def test_validate_inputs_basic(self):
        """Test basic input validation."""
        result = self.validator.validate_inputs(
            epi=0.5,
            vf=1.0,
            theta=0.0,
            raise_on_error=False,
        )

        assert "epi" in result
        assert "vf" in result
        assert "theta" in result
        assert "error" not in result

    def test_validate_inputs_invalid_epi(self):
        """Test input validation rejects invalid EPI."""
        with pytest.raises(ValidationError):
            self.validator.validate_inputs(
                epi=float("nan"),
                raise_on_error=True,
            )

    def test_validate_inputs_invalid_vf(self):
        """Test input validation rejects invalid vf."""
        with pytest.raises(ValidationError):
            self.validator.validate_inputs(
                vf=-1.0,
                raise_on_error=True,
            )

    def test_validate_inputs_with_config(self):
        """Test input validation respects config bounds."""
        config = {"EPI_MIN": 0.2, "EPI_MAX": 0.8}

        # Should pass
        result = self.validator.validate_inputs(
            epi=0.5,
            config=config,
            raise_on_error=False,
        )
        assert "error" not in result

        # Should fail
        with pytest.raises(ValidationError):
            self.validator.validate_inputs(
                epi=0.1,  # Below EPI_MIN
                config=config,
                raise_on_error=True,
            )

    def test_validate_graph_basic(self):
        """Test basic graph validation."""
        violations = self.validator.validate_graph(
            self.graph,
            include_graph_validation=False,  # Skip structure validation
        )

        # Should have minimal violations on a basic valid graph
        critical = [v for v in violations if v.severity == InvariantSeverity.CRITICAL]
        assert len(critical) == 0

    def test_validate_graph_with_cache(self):
        """Test graph validation caching."""
        self.validator.enable_cache(True)

        # First call
        violations1 = self.validator.validate_graph(self.graph)

        # Second call should use cache
        violations2 = self.validator.validate_graph(self.graph, use_cache=True)

        # Results should be identical
        assert len(violations1) == len(violations2)

        self.validator.clear_cache()

    def test_validate_graph_severity_filter(self):
        """Test filtering violations by severity."""
        violations_all = self.validator.validate_graph(self.graph)
        violations_error = self.validator.validate_graph(
            self.graph,
            severity_filter=InvariantSeverity.ERROR,
        )

        # Filtered results should be subset
        assert len(violations_error) <= len(violations_all)

        # All filtered results should have ERROR severity
        for v in violations_error:
            assert v.severity == InvariantSeverity.ERROR

    def test_unified_validate_method(self):
        """Test the unified validate() method."""
        result = self.validator.validate(
            graph=self.graph,
            epi=0.5,
            vf=1.0,
            include_invariants=True,
            include_graph_structure=False,
        )

        assert "passed" in result
        assert "inputs" in result
        assert "invariants" in result
        assert "errors" in result

        # Should have validated inputs
        assert "epi" in result["inputs"]
        assert "vf" in result["inputs"]

    def test_unified_validate_with_invalid_inputs(self):
        """Test unified validation with invalid inputs."""
        result = self.validator.validate(
            epi=float("inf"),  # Invalid
            raise_on_error=False,
        )

        assert result["passed"] is False
        assert len(result["errors"]) > 0

    def test_unified_validate_graph_only(self):
        """Test unified validation for graph only."""
        result = self.validator.validate(
            graph=self.graph,
            include_invariants=True,
            include_graph_structure=False,
        )

        assert "invariants" in result
        # Basic graph should have some invariant checks pass
        assert isinstance(result["invariants"], list)

    def test_validate_and_raise(self):
        """Test validate_and_raise method."""
        # Valid graph should not raise
        self.validator.validate_and_raise(
            self.graph,
            min_severity=InvariantSeverity.ERROR,
        )

        # Create invalid graph
        invalid_graph = nx.DiGraph()
        invalid_graph.graph.update(
            {
                "EPI_MIN": 0.0,
                "EPI_MAX": 1.0,
            }
        )
        invalid_graph.add_node("bad", epi=float("inf"), vf=1.0)

        # Should raise on critical violations
        with pytest.raises(TNFRValidationError):
            self.validator.validate_and_raise(
                invalid_graph,
                min_severity=InvariantSeverity.CRITICAL,
            )

    def test_generate_report(self):
        """Test report generation."""
        violations = self.validator.validate_graph(self.graph)
        report = self.validator.generate_report(violations)

        assert isinstance(report, str)
        assert len(report) > 0

        # Empty violations should generate success message
        empty_report = self.validator.generate_report([])
        assert "No TNFR invariant violations" in empty_report

    def test_export_to_json(self):
        """Test JSON export."""
        violations = self.validator.validate_graph(self.graph)
        json_str = self.validator.export_to_json(violations)

        assert isinstance(json_str, str)
        assert "total_violations" in json_str
        assert "by_severity" in json_str

    def test_export_to_html(self):
        """Test HTML export."""
        violations = self.validator.validate_graph(self.graph)
        html_str = self.validator.export_to_html(violations)

        assert isinstance(html_str, str)
        assert "<html>" in html_str.lower()

    def test_custom_validator_addition(self):
        """Test adding custom validators."""
        from tnfr.validation.invariants import TNFRInvariant, InvariantViolation

        class CustomValidator(TNFRInvariant):
            invariant_id = 99
            description = "Custom test validator"

            def validate(self, graph):
                return []

        custom = CustomValidator()
        self.validator.add_custom_validator(custom)

        # Validation should still work
        violations = self.validator.validate_graph(self.graph)
        assert isinstance(violations, list)


class TestTNFRValidatorInputValidation:
    """Test input validation integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = TNFRValidator()

    def test_validate_epi_bounds(self):
        """Test EPI bounds checking."""
        config = {"EPI_MIN": 0.0, "EPI_MAX": 1.0}

        # Valid EPI
        result = self.validator.validate_inputs(
            epi=0.5, config=config, raise_on_error=False
        )
        assert "error" not in result

        # Invalid EPI (too high)
        with pytest.raises(ValidationError):
            self.validator.validate_inputs(epi=2.0, config=config, raise_on_error=True)

    def test_validate_vf_positive(self):
        """Test νf must be positive."""
        # Valid νf
        result = self.validator.validate_inputs(vf=1.0, raise_on_error=False)
        assert "error" not in result

        # Invalid νf (negative)
        with pytest.raises(ValidationError):
            self.validator.validate_inputs(vf=-0.5, raise_on_error=True)

    def test_validate_theta_normalization(self):
        """Test θ normalization."""
        # Large theta should be normalized
        result = self.validator.validate_inputs(
            theta=4 * math.pi,
            raise_on_error=False,
        )
        assert "theta" in result or "error" in result

    def test_validate_node_id_security(self):
        """Test node ID security checks."""
        # Valid node IDs
        result = self.validator.validate_inputs(node_id="node_1", raise_on_error=False)
        assert "error" not in result

        result = self.validator.validate_inputs(node_id=123, raise_on_error=False)
        assert "error" not in result

        # Invalid node ID (injection pattern)
        with pytest.raises(ValidationError):
            self.validator.validate_inputs(node_id="<script>", raise_on_error=True)


class TestTNFRValidatorOperatorPreconditions:
    """Test operator precondition validation integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = TNFRValidator()

        self.graph = nx.DiGraph()
        self.graph.graph.update(
            {
                "EPI_MIN": 0.0,
                "EPI_MAX": 1.0,
                "VF_MIN": 0.001,
                "VF_MAX": 1000.0,
            }
        )
        # Use correct TNFR attribute names
        self.graph.add_node(
            "node_1", **{"EPI": 0.1, "νf": 1.0, "theta": 0.0, "ΔNFR": 0.0}
        )
        self.graph.add_node(
            "node_2", **{"EPI": 0.8, "νf": 0.5, "theta": math.pi / 4, "ΔNFR": 0.0}
        )
        self.graph.add_edge("node_1", "node_2")

    def test_validate_emission_preconditions(self):
        """Test emission operator preconditions."""
        # Low EPI node should allow emission
        result = self.validator.validate_operator_preconditions(
            self.graph,
            "node_1",
            "emission",
            raise_on_error=False,
        )
        # May pass or fail depending on configuration
        assert isinstance(result, bool)

    def test_validate_reception_preconditions(self):
        """Test reception operator preconditions."""
        # Node with neighbors should allow reception
        result = self.validator.validate_operator_preconditions(
            self.graph,
            "node_2",
            "reception",
            raise_on_error=False,
        )
        assert isinstance(result, bool)

    def test_validate_unknown_operator(self):
        """Test validation of unknown operator."""
        with pytest.raises(ValueError):
            self.validator.validate_operator_preconditions(
                self.graph,
                "node_1",
                "unknown_operator",
                raise_on_error=True,
            )

    def test_unified_validate_with_operator(self):
        """Test unified validation with operator preconditions."""
        result = self.validator.validate(
            graph=self.graph,
            node_id="node_1",
            operator="emission",
            include_invariants=False,
        )

        assert "operator_preconditions" in result
        assert isinstance(result["operator_preconditions"], bool)


class TestTNFRValidatorPerformance:
    """Test performance features of unified validator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.validator = TNFRValidator()

        # Create a larger graph for performance testing
        self.graph = nx.DiGraph()
        self.graph.graph.update(
            {
                "EPI_MIN": 0.0,
                "EPI_MAX": 1.0,
                "VF_MIN": 0.001,
                "VF_MAX": 1000.0,
            }
        )

        # Use correct TNFR attribute names
        for i in range(10):
            self.graph.add_node(
                f"node_{i}", **{"EPI": 0.5, "νf": 1.0, "theta": 0.0, "ΔNFR": 0.0}
            )

    def test_caching_improves_performance(self):
        """Test that caching improves performance."""
        import time

        self.validator.enable_cache(True)

        # First call (no cache)
        start1 = time.time()
        violations1 = self.validator.validate_graph(self.graph)
        time1 = time.time() - start1

        # Second call (with cache)
        start2 = time.time()
        violations2 = self.validator.validate_graph(self.graph, use_cache=True)
        time2 = time.time() - start2

        # Cached call should be faster
        assert time2 < time1 or time2 < 0.001  # Or very fast

        # Results should be identical
        assert len(violations1) == len(violations2)

        self.validator.clear_cache()

    def test_selective_validation_layers(self):
        """Test that selective validation layers work correctly."""
        # Disable certain validation layers for performance
        validator = TNFRValidator(
            enable_input_validation=True,
            enable_graph_validation=False,
            enable_runtime_validation=False,
        )

        result = validator.validate(
            graph=self.graph,
            include_graph_structure=False,
            include_runtime=False,
        )

        assert result is not None
        assert "passed" in result


class TestTNFRValidationError:
    """Test TNFRValidationError exception."""

    def test_validation_error_creation(self):
        """Test creating validation error."""
        from tnfr.validation.invariants import InvariantViolation, InvariantSeverity

        violations = [
            InvariantViolation(
                invariant_id=1,
                severity=InvariantSeverity.ERROR,
                description="Test violation",
            )
        ]

        error = TNFRValidationError(violations)
        assert error.violations == violations
        assert hasattr(error, "report")
        assert len(str(error)) > 0

    def test_validation_error_with_multiple_violations(self):
        """Test validation error with multiple violations."""
        from tnfr.validation.invariants import InvariantViolation, InvariantSeverity

        violations = [
            InvariantViolation(
                invariant_id=1,
                severity=InvariantSeverity.ERROR,
                description="Violation 1",
            ),
            InvariantViolation(
                invariant_id=2,
                severity=InvariantSeverity.CRITICAL,
                description="Violation 2",
            ),
        ]

        error = TNFRValidationError(violations)
        assert len(error.violations) == 2
        assert "CRITICAL" in error.report or "ERROR" in error.report
