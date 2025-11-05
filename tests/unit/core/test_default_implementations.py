"""Tests for default implementations of TNFR interfaces."""

import pytest
import networkx as nx

from tnfr.core.default_implementations import (
    DefaultValidationService,
    DefaultOperatorRegistry,
    DefaultDynamicsEngine,
    DefaultTelemetryCollector,
)
from tnfr.structural import create_nfr
from tnfr.dynamics import set_delta_nfr_hook, dnfr_epi_vf_mixed


def test_default_validation_service_validates_sequence():
    """Verify DefaultValidationService validates sequences correctly."""
    service = DefaultValidationService()

    # Valid canonical sequence should not raise
    service.validate_sequence(["emission", "reception", "coherence", "coupling", "resonance", "silence"])

    # Invalid sequence should raise
    with pytest.raises(ValueError, match="Invalid sequence"):
        service.validate_sequence(["invalid_operator"])


def test_default_validation_service_validates_graph():
    """Verify DefaultValidationService validates graph states."""
    service = DefaultValidationService()
    G, _ = create_nfr("test", epi=1.0, vf=1.0)

    # Valid graph should not raise
    service.validate_graph_state(G)


def test_default_operator_registry_gets_operator():
    """Verify DefaultOperatorRegistry retrieves operators."""
    registry = DefaultOperatorRegistry()

    # Should retrieve known operators
    operator = registry.get_operator("emission")
    assert operator is not None
    assert operator.name == "emission"


def test_default_operator_registry_raises_for_unknown():
    """Verify registry raises for unknown operators."""
    registry = DefaultOperatorRegistry()

    with pytest.raises(KeyError):
        registry.get_operator("unknown_operator_xyz")


def test_default_dynamics_engine_updates_delta_nfr():
    """Verify DefaultDynamicsEngine updates ΔNFR."""
    engine = DefaultDynamicsEngine()
    G, node = create_nfr("test", epi=1.0, vf=1.0)

    # Configure ΔNFR hook
    set_delta_nfr_hook(G, dnfr_epi_vf_mixed)

    # Should not raise
    engine.update_delta_nfr(G)

    # ΔNFR should be updated (check node has dnfr attribute)
    from tnfr.constants import DNFR_PRIMARY
    assert DNFR_PRIMARY in G.nodes[node]


def test_default_dynamics_engine_integrates_nodal_equation():
    """Verify DefaultDynamicsEngine integrates nodal equation."""
    engine = DefaultDynamicsEngine()
    G, node = create_nfr("test", epi=1.0, vf=1.0)

    # Set dt in graph
    G.graph["dt"] = 0.1

    # Should integrate without raising
    engine.integrate_nodal_equation(G)


def test_default_dynamics_engine_coordinates_phase():
    """Verify DefaultDynamicsEngine coordinates phase."""
    engine = DefaultDynamicsEngine()
    G, node = create_nfr("test", epi=1.0, vf=1.0, theta=0.0)

    # Should coordinate without raising
    engine.coordinate_phase_coupling(G)


def test_default_telemetry_collector_computes_coherence():
    """Verify DefaultTelemetryCollector computes coherence."""
    collector = DefaultTelemetryCollector()
    G, _ = create_nfr("test", epi=1.0, vf=1.0)

    coherence = collector.compute_coherence(G)
    assert isinstance(coherence, float)
    assert 0.0 <= coherence <= 1.0


def test_default_telemetry_collector_computes_sense_index():
    """Verify DefaultTelemetryCollector computes sense index."""
    collector = DefaultTelemetryCollector()
    G, _ = create_nfr("test", epi=1.0, vf=1.0)

    si_metrics = collector.compute_sense_index(G)
    assert isinstance(si_metrics, dict)


def test_default_telemetry_trace_context():
    """Verify trace context captures transitions."""
    collector = DefaultTelemetryCollector()
    G, node = create_nfr("test", epi=1.0, vf=1.0)

    with collector.trace_context(G) as tracer:
        pre_state = tracer.capture_state(G)
        assert "coherence" in pre_state
        assert "node_count" in pre_state

        # Simulate state change
        from tnfr.constants import EPI_PRIMARY
        G.nodes[node][EPI_PRIMARY] = 1.5

        post_state = tracer.capture_state(G)
        tracer.record_transition("emission", pre_state, post_state)

    # Check transitions were recorded
    assert "_trace_transitions" in G.graph
    transitions = G.graph["_trace_transitions"]
    assert len(transitions) > 0
    assert transitions[0]["operator"] == "emission"
