"""Core interfaces defining the TNFR architectural contracts.

This module defines Protocol-based interfaces that separate responsibilities
across the TNFR engine layers. Each protocol specifies the minimal contract
required for a component to participate in the structural orchestration without
imposing implementation details.

Protocols (Interfaces)
----------------------
OperatorRegistry
    Maps operator tokens to structural operator implementations.
ValidationService
    Validates sequences and graph states against TNFR invariants.
DynamicsEngine
    Computes ΔNFR, integrates the nodal equation, and coordinates phase.
TelemetryCollector
    Captures coherence metrics, sense index, and structural traces.

Notes
-----
These interfaces use :class:`typing.Protocol` to enable duck typing and
structural subtyping. Implementations do not need to explicitly inherit from
these protocols; they need only provide the specified methods with compatible
signatures.

Examples
--------
Custom validation service implementing ValidationService protocol:

>>> class StrictValidator:
...     def validate_sequence(self, sequence):
...         # Custom validation logic
...         pass
...     def validate_graph_state(self, graph):
...         # Custom graph validation
...         pass
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..types import TNFRGraph
    from ..operators.definitions import Operator

__all__ = (
    "OperatorRegistry",
    "ValidationService",
    "DynamicsEngine",
    "TelemetryCollector",
    "TraceContext",
)


@runtime_checkable
class OperatorRegistry(Protocol):
    """Interface for registering and retrieving structural operators.

    The operator registry maintains the mapping between operator tokens
    (strings or Glyph codes) and their concrete implementations. It ensures
    that only canonical operators are accessible during sequence execution.
    """

    def get_operator(self, token: str) -> Operator:
        """Retrieve operator implementation for the given token.

        Parameters
        ----------
        token : str
            Operator identifier (e.g., "emission", "coherence").

        Returns
        -------
        Operator
            The structural operator implementation.

        Raises
        ------
        KeyError
            When the token does not map to a registered operator.
        """
        ...

    def register_operator(self, operator: Operator) -> None:
        """Register a structural operator implementation.

        Parameters
        ----------
        operator : Operator
            The operator to register in the global registry.
        """
        ...


@runtime_checkable
class ValidationService(Protocol):
    """Interface for validating sequences and graph states.

    The validation service guards TNFR invariants by checking operator
    sequences against grammar rules and ensuring graph states remain within
    canonical bounds (νf, phase, ΔNFR ranges).
    """

    def validate_sequence(self, sequence: list[str]) -> None:
        """Validate an operator sequence against TNFR grammar.

        Parameters
        ----------
        sequence : list of str
            Operator tokens to validate as a trajectory.

        Raises
        ------
        ValueError
            When the sequence violates TNFR grammar rules or operator closure.
        """
        ...

    def validate_graph_state(self, graph: TNFRGraph) -> None:
        """Validate graph state against structural invariants.

        Parameters
        ----------
        graph : TNFRGraph
            Graph whose node attributes (EPI, νf, θ, ΔNFR) are validated.

        Raises
        ------
        ValueError
            When node attributes violate canonical bounds or type constraints.
        """
        ...


@runtime_checkable
class DynamicsEngine(Protocol):
    """Interface for computing ΔNFR and integrating the nodal equation.

    The dynamics engine orchestrates the temporal evolution of nodes by
    computing internal reorganization gradients (ΔNFR), integrating the
    nodal equation ∂EPI/∂t = νf · ΔNFR(t), and coordinating phase coupling
    across the network.
    """

    def update_delta_nfr(self, graph: TNFRGraph) -> None:
        """Compute and update ΔNFR for all nodes in the graph.

        This method implements the canonical ΔNFR computation using
        configured hooks (e.g., dnfr_epi_vf_mixed, dnfr_laplacian).

        Parameters
        ----------
        graph : TNFRGraph
            Graph whose node ΔNFR attributes are updated.
        """
        ...

    def integrate_nodal_equation(self, graph: TNFRGraph) -> None:
        """Integrate the nodal equation to update EPI, νf, and phase.

        This method applies the canonical integrator to advance EPI based on
        the structural frequency and ΔNFR gradient, while optionally
        adapting νf and coordinating phase synchrony.

        Parameters
        ----------
        graph : TNFRGraph
            Graph whose nodes evolve according to ∂EPI/∂t = νf · ΔNFR(t).
        """
        ...

    def coordinate_phase_coupling(self, graph: TNFRGraph) -> None:
        """Coordinate phase synchronization across coupled nodes.

        This method ensures that phase relationships between connected nodes
        respect resonance conditions and structural coupling strength.

        Parameters
        ----------
        graph : TNFRGraph
            Graph whose phase attributes are coordinated.
        """
        ...


@runtime_checkable
class TraceContext(Protocol):
    """Interface for trace context managers used by telemetry collectors."""

    def capture_state(self, graph: TNFRGraph) -> dict[str, Any]:
        """Capture current graph state for telemetry.

        Parameters
        ----------
        graph : TNFRGraph
            Graph whose state is captured.

        Returns
        -------
        dict
            State snapshot including coherence, phase, νf distributions.
        """
        ...

    def record_transition(
        self,
        operator_token: str,
        pre_state: dict[str, Any],
        post_state: dict[str, Any],
    ) -> None:
        """Record operator transition in telemetry traces.

        Parameters
        ----------
        operator_token : str
            The operator that caused the transition.
        pre_state : dict
            State before operator application.
        post_state : dict
            State after operator application.
        """
        ...


@runtime_checkable
class TelemetryCollector(Protocol):
    """Interface for collecting telemetry and structural traces.

    The telemetry collector measures coherence (C(t)), sense index (Si),
    and captures structural traces documenting how operators reorganize
    the network over time.
    """

    def trace_context(self, graph: TNFRGraph) -> TraceContext:
        """Create a trace context for capturing operator effects.

        Parameters
        ----------
        graph : TNFRGraph
            Graph being traced.

        Returns
        -------
        TraceContext
            Context manager for capturing state transitions.
        """
        ...

    def compute_coherence(self, graph: TNFRGraph) -> float:
        """Compute total coherence C(t) across the network.

        Parameters
        ----------
        graph : TNFRGraph
            Graph whose coherence is measured.

        Returns
        -------
        float
            Global coherence value C(t) ∈ [0, 1], where higher values
            indicate greater structural stability.
        """
        ...

    def compute_sense_index(self, graph: TNFRGraph) -> dict[str, Any]:
        """Compute sense index (Si) measuring reorganization capacity.

        Parameters
        ----------
        graph : TNFRGraph
            Graph whose sense index is computed.

        Returns
        -------
        dict
            Sense index metrics including total Si and per-node contributions.
        """
        ...
