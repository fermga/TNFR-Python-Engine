"""Tests for TNFR core interfaces.

These tests verify that the Protocol interfaces are correctly defined and
that implementations satisfy the interface contracts.
"""

import pytest

from tnfr.core.interfaces import (
    DynamicsEngine,
    OperatorRegistry,
    TelemetryCollector,
    TraceContext,
    ValidationService,
)


def test_validation_service_protocol():
    """Verify ValidationService protocol is defined correctly."""
    # Check that protocol has required methods
    assert hasattr(ValidationService, "validate_sequence")
    assert hasattr(ValidationService, "validate_graph_state")


def test_operator_registry_protocol():
    """Verify OperatorRegistry protocol is defined correctly."""
    assert hasattr(OperatorRegistry, "get_operator")
    assert hasattr(OperatorRegistry, "register_operator")


def test_dynamics_engine_protocol():
    """Verify DynamicsEngine protocol is defined correctly."""
    assert hasattr(DynamicsEngine, "update_delta_nfr")
    assert hasattr(DynamicsEngine, "integrate_nodal_equation")
    assert hasattr(DynamicsEngine, "coordinate_phase_coupling")


def test_telemetry_collector_protocol():
    """Verify TelemetryCollector protocol is defined correctly."""
    assert hasattr(TelemetryCollector, "trace_context")
    assert hasattr(TelemetryCollector, "compute_coherence")
    assert hasattr(TelemetryCollector, "compute_sense_index")


def test_trace_context_protocol():
    """Verify TraceContext protocol is defined correctly."""
    assert hasattr(TraceContext, "capture_state")
    assert hasattr(TraceContext, "record_transition")


def test_protocol_duck_typing():
    """Verify protocols work with duck typing (structural subtyping)."""

    # Mock implementation that satisfies ValidationService
    class MockValidator:
        def validate_sequence(self, sequence):
            pass

        def validate_graph_state(self, graph):
            pass

    # Should not raise - duck typing should work
    validator = MockValidator()
    assert isinstance(validator, ValidationService)


def test_protocol_incomplete_implementation():
    """Verify incomplete implementations are not recognized."""

    # Incomplete implementation missing validate_graph_state
    class IncompleteValidator:
        def validate_sequence(self, sequence):
            pass

    validator = IncompleteValidator()
    # Should not be recognized as ValidationService
    assert not isinstance(validator, ValidationService)
