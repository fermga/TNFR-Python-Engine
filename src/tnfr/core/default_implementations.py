"""Default implementations of TNFR core interfaces.

This module provides concrete implementations of the architectural interfaces
that wrap existing TNFR functionality. These implementations maintain backward
compatibility while enabling the new modular architecture.

Classes
-------
DefaultValidationService
    Wraps tnfr.validation for sequence and graph validation.
DefaultOperatorRegistry
    Wraps tnfr.operators.registry for operator management.
DefaultDynamicsEngine
    Wraps tnfr.dynamics for ΔNFR computation and integration.
DefaultTelemetryCollector
    Wraps tnfr.metrics for coherence and sense index computation.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..types import TNFRGraph
    from ..operators.definitions import Operator

__all__ = (
    "DefaultValidationService",
    "DefaultOperatorRegistry",
    "DefaultDynamicsEngine",
    "DefaultTelemetryCollector",
)


class DefaultValidationService:
    """Default implementation of ValidationService using tnfr.validation.

    This implementation wraps the existing validation infrastructure to
    provide the ValidationService interface without modifying existing code.
    """

    def validate_sequence(self, sequence: list[str]) -> None:
        """Validate operator sequence using canonical grammar rules.

        Parameters
        ----------
        sequence : list of str
            Operator tokens to validate.

        Raises
        ------
        ValueError
            When sequence violates TNFR grammar or operator closure.
        """
        from ..validation import validate_sequence as _validate_sequence

        # validate_sequence returns ValidationOutcome
        outcome = _validate_sequence(sequence)
        if not outcome.passed:
            summary_message = outcome.summary.get("message", "validation failed")
            raise ValueError(f"Invalid sequence: {summary_message}")

    def validate_graph_state(self, graph: TNFRGraph) -> None:
        """Validate graph state using canonical validators.

        Parameters
        ----------
        graph : TNFRGraph
            Graph to validate.

        Raises
        ------
        ValueError
            When graph state violates structural invariants.
        """
        from ..validation import run_validators

        # run_validators raises on failure by default
        run_validators(graph)


class DefaultOperatorRegistry:
    """Default implementation of OperatorRegistry using tnfr.operators.

    This implementation wraps the global OPERATORS registry to provide
    the OperatorRegistry interface.
    """

    def get_operator(self, token: str) -> Operator:
        """Retrieve operator by token from global registry.

        Parameters
        ----------
        token : str
            Operator identifier.

        Returns
        -------
        Operator
            The structural operator implementation (class, not instance).

        Raises
        ------
        KeyError
            When token is not registered.
        """
        from ..operators.registry import get_operator_class

        # get_operator_class returns the operator class
        return get_operator_class(token)

    def register_operator(self, operator: Operator) -> None:
        """Register operator in global registry.

        Parameters
        ----------
        operator : Operator
            Operator to register.
        """
        from ..operators.registry import OPERATORS

        # Register by operator name
        OPERATORS[operator.name] = operator.__class__


class DefaultDynamicsEngine:
    """Default implementation of DynamicsEngine using tnfr.dynamics.

    This implementation wraps existing dynamics functions to provide
    the DynamicsEngine interface.
    """

    def update_delta_nfr(self, graph: TNFRGraph) -> None:
        """Compute ΔNFR using configured hook.

        Parameters
        ----------
        graph : TNFRGraph
            Graph to update.
        """
        # Get the configured ΔNFR hook from graph metadata
        compute = graph.graph.get("compute_delta_nfr")
        if callable(compute):
            compute(graph)

    def integrate_nodal_equation(self, graph: TNFRGraph) -> None:
        """Integrate nodal equation to update EPI.

        Parameters
        ----------
        graph : TNFRGraph
            Graph to integrate.
        """
        from ..dynamics.integrators import update_epi_via_nodal_equation

        # Use default integration parameters from graph
        dt = graph.graph.get("dt", 0.1)
        update_epi_via_nodal_equation(graph, dt=dt)

    def coordinate_phase_coupling(self, graph: TNFRGraph) -> None:
        """Coordinate phase synchronization.

        Parameters
        ----------
        graph : TNFRGraph
            Graph whose phase is coordinated.
        """
        from ..dynamics.coordination import coordinate_global_local_phase

        # Coordinate phase using default parameters
        coordinate_global_local_phase(graph)


class DefaultTraceContext:
    """Default trace context for telemetry collection.

    This context manager captures graph state before and after operator
    application, recording transitions for structural analysis.
    """

    def __init__(self, graph: TNFRGraph):
        """Initialize trace context.

        Parameters
        ----------
        graph : TNFRGraph
            Graph being traced.
        """
        self.graph = graph
        self.transitions: list[dict[str, Any]] = []

    def __enter__(self):
        """Enter trace context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit trace context."""
        # Save transitions when exiting context
        self._save_transitions()
        return False

    def _save_transitions(self) -> None:
        """Save transitions to graph metadata."""
        if self.transitions:
            existing = self.graph.graph.get("_trace_transitions", [])
            existing_copy = list(existing)  # Make a copy to avoid mutation
            existing_copy.extend(self.transitions)
            self.graph.graph["_trace_transitions"] = existing_copy

    def capture_state(self, graph: TNFRGraph) -> dict[str, Any]:
        """Capture current graph state.

        Parameters
        ----------
        graph : TNFRGraph
            Graph to capture.

        Returns
        -------
        dict
            State snapshot.
        """
        from ..metrics.common import compute_coherence

        # Capture key metrics
        return {
            "coherence": compute_coherence(graph),
            "node_count": graph.number_of_nodes(),
            "edge_count": (graph.number_of_edges() if hasattr(graph, "number_of_edges") else 0),
        }

    def record_transition(
        self,
        operator_token: str,
        pre_state: dict[str, Any],
        post_state: dict[str, Any],
    ) -> None:
        """Record operator transition.

        Parameters
        ----------
        operator_token : str
            Operator that caused transition.
        pre_state : dict
            State before operator.
        post_state : dict
            State after operator.
        """
        self.transitions.append(
            {
                "operator": operator_token,
                "pre": pre_state,
                "post": post_state,
                "delta_coherence": post_state["coherence"] - pre_state["coherence"],
            }
        )


class DefaultTelemetryCollector:
    """Default implementation of TelemetryCollector using tnfr.metrics.

    This implementation wraps existing metrics functions to provide
    the TelemetryCollector interface.
    """

    @contextmanager
    def trace_context(self, graph: TNFRGraph):
        """Create trace context for operator execution.

        Parameters
        ----------
        graph : TNFRGraph
            Graph being traced.

        Yields
        ------
        DefaultTraceContext
            Context for capturing transitions.
        """
        context = DefaultTraceContext(graph)
        try:
            yield context
        finally:
            # Ensure transitions are saved using the helper method
            context._save_transitions()

    def compute_coherence(self, graph: TNFRGraph) -> float:
        """Compute global coherence C(t).

        Parameters
        ----------
        graph : TNFRGraph
            Graph to measure.

        Returns
        -------
        float
            Coherence value in [0, 1].
        """
        from ..metrics.common import compute_coherence

        return compute_coherence(graph)

    def compute_sense_index(self, graph: TNFRGraph) -> dict[str, Any]:
        """Compute sense index Si.

        Parameters
        ----------
        graph : TNFRGraph
            Graph to measure.

        Returns
        -------
        dict
            Sense index metrics.
        """
        from ..metrics.sense_index import compute_Si

        result = compute_Si(graph)

        # Ensure we return a dict
        if isinstance(result, dict):
            return result
        else:
            # If compute_Si returns a scalar, wrap it
            return {"Si": result}
