"""Service orchestration layer for TNFR engine.

This module provides the TNFROrchestrator service that coordinates execution
of operator sequences with clear separation of responsibilities:

1. Validation (delegated to ValidationService)
2. Execution (coordinated through OperatorRegistry)
3. Dynamics (delegated to DynamicsEngine)
4. Telemetry (delegated to TelemetryCollector)

The orchestrator maintains the nodal equation ∂EPI/∂t = νf · ΔNFR(t) while
ensuring that each layer operates independently and can be replaced without
affecting the others.

Examples
--------
Execute a sequence with default services:

>>> from tnfr.core.container import TNFRContainer
>>> from tnfr.services.orchestrator import TNFROrchestrator
>>> from tnfr.structural import create_nfr
>>>
>>> container = TNFRContainer.create_default()
>>> orchestrator = TNFROrchestrator.from_container(container)
>>> G, node = create_nfr("seed", epi=1.0, vf=2.0)
>>> orchestrator.execute_sequence(G, node, ["emission", "coherence"])

Execute with custom services:

>>> class CustomValidator:
...     def validate_sequence(self, seq):
...         print(f"Validating {seq}")
...     def validate_graph_state(self, graph):
...         pass
>>>
>>> container = TNFRContainer()
>>> container.register_singleton(ValidationService, CustomValidator())
>>> # ... register other services
>>> orchestrator = TNFROrchestrator.from_container(container)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from ..core.interfaces import (
        DynamicsEngine,
        OperatorRegistry,
        TelemetryCollector,
        ValidationService,
    )
    from ..types import NodeId, TNFRGraph
    from ..operators.definitions import Operator

__all__ = ("TNFROrchestrator",)


class TNFROrchestrator:
    """Orchestrates TNFR sequence execution with separated responsibilities.

    The orchestrator coordinates validation, execution, dynamics updates, and
    telemetry collection without directly implementing any of these concerns.
    Each responsibility is delegated to a specialized service, enabling
    flexible composition and testing.

    Attributes
    ----------
    _validator : ValidationService
        Service for validating sequences and graph states.
    _registry : OperatorRegistry
        Service for retrieving operator implementations.
    _dynamics : DynamicsEngine
        Service for computing ΔNFR and integrating nodal equation.
    _telemetry : TelemetryCollector
        Service for collecting metrics and traces.

    Examples
    --------
    Create orchestrator with dependency injection:

    >>> from tnfr.core.container import TNFRContainer
    >>> container = TNFRContainer.create_default()
    >>> orch = TNFROrchestrator.from_container(container)

    Execute a validated sequence:

    >>> from tnfr.structural import create_nfr
    >>> G, node = create_nfr("test", epi=1.0, vf=1.0)
    >>> orch.execute_sequence(G, node, ["emission", "coherence"])
    """

    def __init__(
        self,
        validator: ValidationService,
        registry: OperatorRegistry,
        dynamics: DynamicsEngine,
        telemetry: TelemetryCollector,
    ):
        """Initialize orchestrator with injected services.

        Parameters
        ----------
        validator : ValidationService
            Service for validation.
        registry : OperatorRegistry
            Service for operator lookup.
        dynamics : DynamicsEngine
            Service for dynamics computation.
        telemetry : TelemetryCollector
            Service for telemetry collection.
        """
        self._validator = validator
        self._registry = registry
        self._dynamics = dynamics
        self._telemetry = telemetry

    @classmethod
    def from_container(cls, container) -> TNFROrchestrator:
        """Create orchestrator from dependency injection container.

        Parameters
        ----------
        container : TNFRContainer
            Container with registered services.

        Returns
        -------
        TNFROrchestrator
            Configured orchestrator instance.

        Examples
        --------
        >>> from tnfr.core.container import TNFRContainer
        >>> container = TNFRContainer.create_default()
        >>> orch = TNFROrchestrator.from_container(container)
        """
        from ..core.interfaces import (
            DynamicsEngine,
            OperatorRegistry,
            TelemetryCollector,
            ValidationService,
        )

        return cls(
            validator=container.get(ValidationService),
            registry=container.get(OperatorRegistry),
            dynamics=container.get(DynamicsEngine),
            telemetry=container.get(TelemetryCollector),
        )

    def execute_sequence(
        self,
        graph: TNFRGraph,
        node: NodeId,
        sequence: Iterable[str | Operator],
        *,
        enable_telemetry: bool = False,
    ) -> None:
        """Execute operator sequence with separated responsibilities.

        This method coordinates the full execution pipeline:
        1. Validate sequence against TNFR grammar
        2. Convert tokens to operator instances
        3. Apply each operator with telemetry (optional)
        4. Update ΔNFR after each operator
        5. Integrate nodal equation

        Parameters
        ----------
        graph : TNFRGraph
            Graph containing the target node.
        node : NodeId
            Node identifier to receive operators.
        sequence : iterable of str or Operator
            Operator tokens or instances to apply.
        enable_telemetry : bool, optional
            Whether to capture detailed telemetry traces. Default is False.

        Raises
        ------
        ValueError
            When sequence validation fails or operators are unknown.

        Examples
        --------
        Execute with string tokens:

        >>> from tnfr.structural import create_nfr
        >>> G, n = create_nfr("node1", epi=1.0)
        >>> orch = TNFROrchestrator.from_container(container)
        >>> orch.execute_sequence(G, n, ["emission", "coherence"])

        Execute with telemetry enabled:

        >>> orch.execute_sequence(G, n, ["emission"], enable_telemetry=True)
        >>> # Check transitions in G.graph["_trace_transitions"]
        """
        # Convert sequence to list and extract operator tokens
        ops_list = list(sequence)

        # Separate tokens from Operator instances
        tokens = []
        operator_instances = []
        for item in ops_list:
            if isinstance(item, str):
                tokens.append(item)
                operator_instances.append(None)  # Will resolve later
            else:
                # Assume it's an Operator instance
                tokens.append(item.name)
                operator_instances.append(item)

        # Step 1: Validation (delegated to ValidationService)
        if tokens:  # Skip validation for empty sequences
            self._validator.validate_sequence(tokens)

        # Step 2: Execution with optional telemetry
        if enable_telemetry:
            with self._telemetry.trace_context(graph) as tracer:
                self._execute_with_telemetry(graph, node, tokens, operator_instances, tracer)
        else:
            self._execute_without_telemetry(graph, node, tokens, operator_instances)

    def _execute_with_telemetry(self, graph, node, tokens, operator_instances, tracer) -> None:
        """Execute sequence with telemetry capture."""
        for token, op_instance in zip(tokens, operator_instances):
            # Pre-execution: capture state
            pre_state = tracer.capture_state(graph)

            # Execution: apply operator
            if op_instance is None:
                operator_cls = self._registry.get_operator(token)
                operator_cls()(graph, node)  # Instantiate and call
            else:
                op_instance(graph, node)

            # Post-execution: update dynamics
            self._dynamics.update_delta_nfr(graph)

            # Telemetry: record transition
            post_state = tracer.capture_state(graph)
            tracer.record_transition(token, pre_state, post_state)

    def _execute_without_telemetry(self, graph, node, tokens, operator_instances) -> None:
        """Execute sequence without telemetry overhead."""
        for token, op_instance in zip(tokens, operator_instances):
            # Execution: apply operator
            if op_instance is None:
                operator_cls = self._registry.get_operator(token)
                operator_cls()(graph, node)  # Instantiate and call
            else:
                op_instance(graph, node)

            # Post-execution: update dynamics
            self._dynamics.update_delta_nfr(graph)

    def validate_only(self, sequence: list[str]) -> None:
        """Validate sequence without executing.

        This method is useful for pre-flight checks before committing to
        execution.

        Parameters
        ----------
        sequence : list of str
            Operator tokens to validate.

        Raises
        ------
        ValueError
            When sequence violates grammar rules.

        Examples
        --------
        >>> orch.validate_only(["emission", "coherence"])  # OK
        >>> orch.validate_only(["unknown_op"])  # Raises ValueError
        """
        self._validator.validate_sequence(sequence)

    def get_coherence(self, graph: TNFRGraph) -> float:
        """Get current coherence C(t) from telemetry service.

        Parameters
        ----------
        graph : TNFRGraph
            Graph to measure.

        Returns
        -------
        float
            Current coherence value.

        Examples
        --------
        >>> coherence = orch.get_coherence(G)
        >>> print(f"C(t) = {coherence:.3f}")
        """
        return self._telemetry.compute_coherence(graph)

    def get_sense_index(self, graph: TNFRGraph) -> dict:
        """Get sense index Si from telemetry service.

        Parameters
        ----------
        graph : TNFRGraph
            Graph to measure.

        Returns
        -------
        dict
            Sense index metrics.

        Examples
        --------
        >>> si_metrics = orch.get_sense_index(G)
        >>> print(f"Si = {si_metrics['Si']:.3f}")
        """
        return self._telemetry.compute_sense_index(graph)
