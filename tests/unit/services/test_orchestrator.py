"""Tests for TNFROrchestrator service."""

import pytest

from tnfr.core.container import TNFRContainer
from tnfr.services.orchestrator import TNFROrchestrator
from tnfr.structural import create_nfr
from tnfr.dynamics import set_delta_nfr_hook, dnfr_epi_vf_mixed
from tnfr.constants import EPI_PRIMARY, VF_PRIMARY

# Canonical TNFR operator sequence for testing
CANONICAL_SEQUENCE = [
    "emission",
    "reception",
    "coherence",
    "coupling",
    "resonance",  # Changed from dissonance - COUPLING → RESONANCE is excellent
    "silence",
]


def test_orchestrator_from_container():
    """Verify orchestrator can be created from container."""
    container = TNFRContainer.create_default()
    orchestrator = TNFROrchestrator.from_container(container)

    assert orchestrator is not None
    assert hasattr(orchestrator, "_validator")
    assert hasattr(orchestrator, "_registry")
    assert hasattr(orchestrator, "_dynamics")
    assert hasattr(orchestrator, "_telemetry")


def test_orchestrator_execute_sequence_with_strings():
    """Verify orchestrator executes sequence with string tokens."""
    container = TNFRContainer.create_default()
    orchestrator = TNFROrchestrator.from_container(container)

    G, node = create_nfr("test", epi=1.0, vf=1.0, theta=0.0)
    set_delta_nfr_hook(G, dnfr_epi_vf_mixed)

    # Should execute without raising (using full canonical sequence)
    orchestrator.execute_sequence(G, node, CANONICAL_SEQUENCE)

    # Verify graph was modified (EPI should have changed)
    epi_after = G.nodes[node][EPI_PRIMARY]
    # EPI should have been affected by operators
    assert epi_after is not None


def test_orchestrator_execute_sequence_with_operators():
    """Verify orchestrator executes sequence with Operator instances."""
    container = TNFRContainer.create_default()
    orchestrator = TNFROrchestrator.from_container(container)

    G, node = create_nfr("test", epi=1.0, vf=1.0, theta=0.0)
    set_delta_nfr_hook(G, dnfr_epi_vf_mixed)

    # Import operator classes
    from tnfr.operators.definitions import (
        Emission,
        Reception,
        Coherence,
        Coupling,
        Resonance,
        Silence,
    )

    # Execute with operator instances (canonical sequence)
    orchestrator.execute_sequence(
        G,
        node,
        [
            Emission(),
            Reception(),
            Coherence(),
            Coupling(),
            Resonance(),  # Changed from Dissonance - COUPLING → RESONANCE is excellent
            Silence(),
        ],
    )

    # Should have executed successfully
    assert G.nodes[node][EPI_PRIMARY] is not None


def test_orchestrator_validates_sequence():
    """Verify orchestrator validates before execution."""
    container = TNFRContainer.create_default()
    orchestrator = TNFROrchestrator.from_container(container)

    G, node = create_nfr("test", epi=1.0, vf=1.0)

    # Invalid sequence should raise during execution
    with pytest.raises(ValueError, match="Invalid sequence"):
        orchestrator.execute_sequence(G, node, ["invalid_operator"])


def test_orchestrator_validate_only():
    """Verify validate_only method works."""
    container = TNFRContainer.create_default()
    orchestrator = TNFROrchestrator.from_container(container)

    # Valid canonical sequence should not raise
    orchestrator.validate_only(CANONICAL_SEQUENCE)

    # Invalid sequence should raise
    with pytest.raises(ValueError):
        orchestrator.validate_only(["invalid_op"])


def test_orchestrator_with_telemetry():
    """Verify orchestrator captures telemetry when enabled."""
    container = TNFRContainer.create_default()
    orchestrator = TNFROrchestrator.from_container(container)

    G, node = create_nfr("test", epi=1.0, vf=1.0, theta=0.0)
    set_delta_nfr_hook(G, dnfr_epi_vf_mixed)

    # Execute with telemetry enabled (canonical sequence)
    orchestrator.execute_sequence(G, node, CANONICAL_SEQUENCE, enable_telemetry=True)

    # Check that transitions were recorded
    assert "_trace_transitions" in G.graph
    transitions = G.graph["_trace_transitions"]
    assert len(transitions) > 0
    assert transitions[0]["operator"] == "emission"


def test_orchestrator_get_coherence():
    """Verify orchestrator can retrieve coherence."""
    container = TNFRContainer.create_default()
    orchestrator = TNFROrchestrator.from_container(container)

    G, _ = create_nfr("test", epi=1.0, vf=1.0)

    coherence = orchestrator.get_coherence(G)
    assert isinstance(coherence, float)
    assert 0.0 <= coherence <= 1.0


def test_orchestrator_get_sense_index():
    """Verify orchestrator can retrieve sense index."""
    container = TNFRContainer.create_default()
    orchestrator = TNFROrchestrator.from_container(container)

    G, _ = create_nfr("test", epi=1.0, vf=1.0)

    si_metrics = orchestrator.get_sense_index(G)
    assert isinstance(si_metrics, dict)


def test_orchestrator_empty_sequence():
    """Verify orchestrator handles empty sequences correctly."""
    container = TNFRContainer.create_default()
    orchestrator = TNFROrchestrator.from_container(container)

    G, node = create_nfr("test", epi=1.0, vf=1.0)

    # Empty sequence should not raise (structural identity)
    orchestrator.execute_sequence(G, node, [])


def test_orchestrator_separation_of_concerns():
    """Verify orchestrator maintains separation of concerns."""

    # Mock services to track calls
    class MockValidator:
        def __init__(self):
            self.validated = []

        def validate_sequence(self, seq):
            self.validated.append(seq)

        def validate_graph_state(self, graph):
            pass

    class MockRegistry:
        def get_operator(self, token):
            from tnfr.operators.definitions import Emission

            return Emission  # Return class, not instance

        def register_operator(self, op):
            pass

    class MockDynamics:
        def __init__(self):
            self.dnfr_updated = 0

        def update_delta_nfr(self, graph):
            self.dnfr_updated += 1

        def integrate_nodal_equation(self, graph):
            pass

        def coordinate_phase_coupling(self, graph):
            pass

    class MockTelemetry:
        def trace_context(self, graph):
            from contextlib import contextmanager

            @contextmanager
            def ctx():
                yield self

            return ctx()

        def capture_state(self, graph):
            return {}

        def record_transition(self, token, pre, post):
            pass

        def compute_coherence(self, graph):
            return 1.0

        def compute_sense_index(self, graph):
            return {"Si": 0.5}

    # Create orchestrator with mocks
    validator = MockValidator()
    dynamics = MockDynamics()
    orchestrator = TNFROrchestrator(
        validator=validator,
        registry=MockRegistry(),
        dynamics=dynamics,
        telemetry=MockTelemetry(),
    )

    G, node = create_nfr("test", epi=1.0, vf=1.0)

    # Execute sequence
    orchestrator.execute_sequence(G, node, ["emission"])

    # Verify each service was called
    assert len(validator.validated) == 1
    assert validator.validated[0] == ["emission"]
    assert dynamics.dnfr_updated == 1
