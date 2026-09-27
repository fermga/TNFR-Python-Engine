"""Orchestrate configured input, graph and operator validation checks.

The pipeline combines shared input adapters, selected graph/runtime checks,
operator preconditions and ten legacy invariant diagnostics. A successful
report covers the requested checks, not every nodal model obligation or a
certificate of trajectory validity, physical correctness or input security.
"""

from __future__ import annotations

from copy import deepcopy
from html import escape
from typing import Any, Mapping

from ..errors import TNFRValueError
from ..types import NodeId, TNFRGraph
from .invariants import (
    Invariant1_EPIOnlyThroughOperators,
    Invariant2_VfInHzStr,
    Invariant3_DNFRSemantics,
    Invariant4_OperatorClosure,
    Invariant5_ExplicitPhaseChecks,
    Invariant6_NodeBirthCollapse,
    Invariant7_OperationalFractality,
    Invariant8_ControlledDeterminism,
    Invariant9_StructuralMetrics,
    Invariant10_DomainNeutrality,
    InvariantSeverity,
    InvariantViolation,
    TNFRInvariant,
)

__all__ = [
    "TNFRValidator",
    "TNFRValidationError",
]


def _require_boolean_flag(value: Any, name: str) -> None:
    """Reject truthy substitutes before selecting checks or runtime mutation."""
    if not isinstance(value, bool):
        raise TNFRValueError(f"{name} must be a boolean")


def _operator_precondition_validator(graph: TNFRGraph, node: NodeId, operator: str):
    """Resolve existing preconditions after admitting their invocation context."""
    from ..config.operator_names import CANONICAL_OPERATOR_NAMES
    from ..operators import preconditions
    from .input_validation import validate_node_id, validate_tnfr_graph

    if (
        not isinstance(operator, str)
        or operator.lower() not in CANONICAL_OPERATOR_NAMES
    ):
        raise TNFRValueError(
            f"Unknown operator: {operator}",
            context={
                "operator": operator,
                "available": sorted(CANONICAL_OPERATOR_NAMES),
            },
            suggestion="Use a valid canonical operator name.",
        )
    validate_tnfr_graph(graph)
    validate_node_id(node)
    if node not in graph.nodes:
        raise TNFRValueError(f"Operator target node {node!r} is not in graph")
    return getattr(preconditions, f"validate_{operator.lower()}")


class TNFRValidator:
    """Unified TNFR Validation Pipeline.

    This class orchestrates selected validation owners and reports their results.
    Input admission, invariant diagnostics and operator preconditions have
    distinct scopes; omitted checks supply no evidence of success.

    Features
    --------
    - Applies 10 legacy checks covering the six canonical TNFR invariants
    - Shared scalar, identifier and graph-interface input validation
    - Graph structure and coherence validation
    - Runtime canonical validation
    - Operator precondition checking
    - Comprehensive reporting (text, JSON, HTML)
    - Fresh evaluation of live graph checks

    Examples
    --------
    >>> validator = TNFRValidator()
    >>> violations = validator.validate_graph(graph)
    >>> if violations:
    ...     print(validator.generate_report(violations))

    >>> # Validate inputs before operator application
    >>> validator.validate_inputs(epi=0.5, vf=1.0, theta=0.0, config=G.graph)

    >>> # Validate operator preconditions
    >>> validator.validate_operator_preconditions(G, node, "emission")
    """

    def __init__(
        self,
        phase_coupling_threshold: float | None = None,
        enable_input_validation: bool = True,
        enable_graph_validation: bool = True,
        enable_runtime_validation: bool = True,
    ) -> None:
        """Initialize unified TNFR validator.

        Parameters
        ----------
        phase_coupling_threshold : float, optional
            Threshold for phase difference in coupled nodes (default: π/2).
        enable_input_validation : bool, optional
            Enable input validation checks (default: True).
        enable_graph_validation : bool, optional
            Enable graph structure validation (default: True).
        enable_runtime_validation : bool, optional
            Enable runtime canonical validation (default: True).
        """
        for name, value in (
            ("enable_input_validation", enable_input_validation),
            ("enable_graph_validation", enable_graph_validation),
            ("enable_runtime_validation", enable_runtime_validation),
        ):
            _require_boolean_flag(value, name)
        # Initialize core invariant validators
        self._invariant_validators: list[TNFRInvariant] = [
            Invariant1_EPIOnlyThroughOperators(),
            Invariant2_VfInHzStr(),
            Invariant3_DNFRSemantics(),
            Invariant4_OperatorClosure(),
            Invariant6_NodeBirthCollapse(),
            Invariant7_OperationalFractality(),
            Invariant8_ControlledDeterminism(),
            Invariant9_StructuralMetrics(),
            Invariant10_DomainNeutrality(),
        ]

        # Initialize phase validator with custom threshold if provided
        if phase_coupling_threshold is not None:
            self._invariant_validators.append(
                Invariant5_ExplicitPhaseChecks(phase_coupling_threshold)
            )
        else:
            self._invariant_validators.append(Invariant5_ExplicitPhaseChecks())

        self._custom_validators: list[TNFRInvariant] = []

        # Validation pipeline configuration
        self._enable_input_validation = enable_input_validation
        self._enable_graph_validation = enable_graph_validation
        self._enable_runtime_validation = enable_runtime_validation

    def add_custom_validator(self, validator: TNFRInvariant) -> None:
        """Add custom invariant validator.

        Parameters
        ----------
        validator : TNFRInvariant
            Custom validator implementing TNFRInvariant interface.
        """
        self._custom_validators.append(validator)

    def enable_cache(self, enabled: bool = True) -> None:
        """Retain the compatibility switch without caching live graph evidence.

        Graph identity cannot capture mutable state, check selection or custom
        validator dependencies. Every graph invocation therefore evaluates its
        requested checks afresh, regardless of this Boolean argument.
        """
        _require_boolean_flag(enabled, "enabled")

    def clear_cache(self) -> None:
        """Compatibility no-op: this orchestrator retains no graph result cache."""

    def validate(
        self,
        graph: TNFRGraph | None = None,
        *,
        epi: Any = None,
        vf: Any = None,
        theta: Any = None,
        dnfr: Any = None,
        node_id: NodeId | None = None,
        operator: str | None = None,
        include_invariants: bool = True,
        include_graph_structure: bool = True,
        include_runtime: bool = False,
        raise_on_error: bool = False,
    ) -> dict[str, Any]:
        """Run the requested input, graph, invariant and operator checks.

        Flags must be booleans. Supplying an operator requires both a graph
        and a target node. Runtime validation is an opt-in mutating clamp pass,
        including when a later check fails; this pipeline is not transactional.

        Parameters
        ----------
        graph : TNFRGraph, optional
            Graph to validate (required for graph/invariant validation).
        epi : Any, optional
            EPI value to validate.
        vf : Any, optional
            Structural frequency (νf) to validate.
        theta : Any, optional
            Phase (θ) to validate.
        dnfr : Any, optional
            ΔNFR value to validate.
        node_id : NodeId, optional
            Node ID to validate (required for operator preconditions).
        operator : str, optional
            Operator name to validate preconditions for.
        include_invariants : bool, optional
            Include invariant validation (default: True).
        include_graph_structure : bool, optional
            Include graph structure validation (default: True).
        include_runtime : bool, optional
            Include the mutating runtime clamp/validation pass (default: False).
        raise_on_error : bool, optional
            Whether to raise on first error (default: False).

        Returns
        -------
        dict[str, Any]
            Comprehensive validation results including:
            - 'passed': bool - Overall validation status
            - 'inputs': dict - Input validation results
            - 'graph_structure': dict - Graph structure validation results
            - 'runtime': dict - Runtime validation results
            - 'invariants': list - Invariant violations
            - 'operator_preconditions': bool - Operator precondition status
            - 'errors': list - Any errors encountered

        Examples
        --------
        >>> validator = TNFRValidator()
        >>> # Validate graph with inputs
        >>> result = validator.validate(
        ...     graph=G,
        ...     epi=0.5,
        ...     vf=1.0,
        ...     include_invariants=True
        ... )
        >>> if not result['passed']:
        ...     print(f"Validation failed: {result['errors']}")

        >>> # Validate operator preconditions
        >>> result = validator.validate(
        ...     graph=G,
        ...     node_id="node_1",
        ...     operator="emission"
        ... )
        >>> if result['operator_preconditions']:
        ...     # Apply operator
        ...     pass
        """
        _require_boolean_flag(raise_on_error, "raise_on_error")
        results: dict[str, Any] = {
            "passed": True,
            "inputs": {},
            "graph_structure": None,
            "runtime": None,
            "invariants": [],
            "operator_preconditions": None,
            "errors": [],
        }

        from .input_validation import validate_tnfr_graph

        try:
            for name, value in (
                ("include_invariants", include_invariants),
                ("include_graph_structure", include_graph_structure),
                ("include_runtime", include_runtime),
            ):
                _require_boolean_flag(value, name)
            if graph is not None:
                validate_tnfr_graph(graph)
            if operator is not None and (graph is None or node_id is None):
                raise TNFRValueError("Operator validation requires graph and node_id")
            if operator is not None:
                _operator_precondition_validator(graph, node_id, operator)
        except Exception as exc:
            if raise_on_error:
                raise
            results["passed"] = False
            results["errors"].append(f"Validation request: {exc}")
            return results

        # Input validation
        if any(value is not None for value in (epi, vf, theta, dnfr, node_id)):
            try:
                results["inputs"] = self.validate_inputs(
                    epi=epi,
                    vf=vf,
                    theta=theta,
                    dnfr=dnfr,
                    node_id=node_id,
                    raise_on_error=raise_on_error,
                )
                if "error" in results["inputs"]:
                    results["passed"] = False
                    results["errors"].append(
                        f"Input validation: {results['inputs']['error']}"
                    )
            except Exception as e:
                results["passed"] = False
                results["errors"].append(f"Input validation failed: {str(e)}")
                if raise_on_error:
                    raise

        # Graph validation
        if graph is not None:
            # Graph structure validation
            if include_graph_structure:
                try:
                    results["graph_structure"] = self.validate_graph_structure(
                        graph,
                        raise_on_error=raise_on_error,
                    )
                    if not results["graph_structure"].get("passed", False):
                        results["passed"] = False
                        results["errors"].append(
                            f"Graph structure: {results['graph_structure'].get('error', 'Failed')}"
                        )
                except Exception as e:
                    results["passed"] = False
                    results["errors"].append(
                        f"Graph structure validation failed: {str(e)}"
                    )
                    if raise_on_error:
                        raise

            # Runtime canonical validation
            if include_runtime:
                try:
                    results["runtime"] = self.validate_runtime_canonical(
                        graph,
                        raise_on_error=raise_on_error,
                    )
                    if not results["runtime"].get("passed", False):
                        results["passed"] = False
                        results["errors"].append(
                            f"Runtime validation: {results['runtime'].get('error', 'Failed')}"
                        )
                except Exception as e:
                    results["passed"] = False
                    results["errors"].append(f"Runtime validation failed: {str(e)}")
                    if raise_on_error:
                        raise

            # Invariant validation
            if include_invariants:
                try:
                    violations = self.validate_graph(
                        graph,
                        include_graph_validation=False,  # Already done above
                        include_runtime_validation=False,  # Already done above
                    )
                    results["invariants"] = violations
                    if violations:
                        # Check if there are any ERROR or CRITICAL violations
                        critical_violations = [
                            v
                            for v in violations
                            if v.severity
                            in (InvariantSeverity.ERROR, InvariantSeverity.CRITICAL)
                        ]
                        if critical_violations:
                            results["passed"] = False
                            results["errors"].append(
                                f"{len(critical_violations)} critical invariant violations found"
                            )
                            if raise_on_error:
                                raise TNFRValidationError(critical_violations)
                except Exception as e:
                    results["passed"] = False
                    results["errors"].append(f"Invariant validation failed: {str(e)}")
                    if raise_on_error:
                        raise

            # Operator preconditions validation
            if operator is not None and node_id is not None:
                try:
                    results["operator_preconditions"] = (
                        self.validate_operator_preconditions(
                            graph,
                            node_id,
                            operator,
                            raise_on_error=raise_on_error,
                        )
                    )
                    if not results["operator_preconditions"]:
                        results["passed"] = False
                        results["errors"].append(
                            f"Operator '{operator}' preconditions not met for node {node_id}"
                        )
                except Exception as e:
                    results["passed"] = False
                    results["errors"].append(
                        f"Operator precondition validation failed: {str(e)}"
                    )
                    if raise_on_error:
                        raise

        return results

    def validate_inputs(
        self,
        *,
        epi: Any = None,
        vf: Any = None,
        theta: Any = None,
        dnfr: Any = None,
        node_id: Any = None,
        glyph: Any = None,
        graph: Any = None,
        config: Mapping[str, Any] | None = None,
        raise_on_error: bool = True,
    ) -> dict[str, Any]:
        """Validate structural operator inputs.

        This adapter returns normalized values from the shared input helpers.
        It does not certify graph dynamics or operator preconditions. ``None``
        means an omitted argument; disabled input validation returns an empty dict.

        Parameters
        ----------
        epi : Any, optional
            Finite signed scalar or uniform-real EPI value to validate.
        vf : Any, optional
            νf (structural frequency) value to validate.
        theta : Any, optional
            θ (phase) value to validate.
        dnfr : Any, optional
            Finite represented-real ΔNFR pressure value to validate.
        node_id : Any, optional
            Node identifier to validate.
        glyph : Any, optional
            Glyph enumeration to validate.
        graph : Any, optional
            Object to check for the required graph interface. Full graph
            structure and invariant validation are separate operations.
        config : Mapping[str, Any], optional
            Reserved compatibility argument, currently not consumed. Graph
            configuration keys do not override input bounds. Frequency policy
            and phase normalization use the shared input validator's config.
        raise_on_error : bool, optional
            Whether to raise exception on validation failure (default: True).

        Returns
        -------
        dict[str, Any]
            Supplied parameter names mapped to normalized values. On failure
            with ``raise_on_error=False``, retain preceding validated values
            and add ``error`` with the first failure; later inputs are unchecked.

        Raises
        ------
        ValidationError
            If any validation fails and raise_on_error is True.

        Examples
        --------
        >>> validator = TNFRValidator()
        >>> validator.validate_inputs(epi=0.5, vf=1.0, theta=0.0)
        {'epi': 0.5, 'vf': 1.0, 'theta': 0.0}
        """
        _require_boolean_flag(raise_on_error, "raise_on_error")
        if not self._enable_input_validation:
            return {}

        from .input_validation import (
            ValidationError,
            validate_dnfr_value,
            validate_epi_value,
            validate_glyph,
            validate_node_id,
            validate_theta_value,
            validate_tnfr_graph,
            validate_vf_value,
        )

        results: dict[str, Any] = {}
        for name, value, validate_value in (
            ("epi", epi, validate_epi_value),
            ("vf", vf, validate_vf_value),
            ("theta", theta, validate_theta_value),
            ("dnfr", dnfr, validate_dnfr_value),
            ("node_id", node_id, validate_node_id),
            ("glyph", glyph, validate_glyph),
            ("graph", graph, validate_tnfr_graph),
        ):
            if value is None:
                continue
            try:
                results[name] = validate_value(value)
            except ValidationError as exc:
                if raise_on_error:
                    raise
                results["error"] = str(exc)
                break
        return results

    def validate_operator_preconditions(
        self,
        graph: TNFRGraph,
        node: NodeId,
        operator: str,
        raise_on_error: bool = True,
    ) -> bool:
        """Validate operator preconditions before application.

        Delegate to the existing named precondition owner. This checks neither
        complete word grammar nor execution postconditions; owners may retain
        their documented telemetry effects.

        Parameters
        ----------
        graph : TNFRGraph
            Graph containing the target node.
        node : NodeId
            Target node for operator application.
        operator : str
            Name of the operator to validate (e.g., "emission", "coherence").
        raise_on_error : bool, optional
            Whether to raise exception on failure (default: True).

        Returns
        -------
        bool
            True if preconditions are met, False otherwise.

        Raises
        ------
        OperatorPreconditionError
            If preconditions are not met and raise_on_error is True.

        Examples
        --------
        >>> validator = TNFRValidator()
        >>> if validator.validate_operator_preconditions(G, node, "emission"):
        ...     # Apply emission operator
        ...     pass
        """
        _require_boolean_flag(raise_on_error, "raise_on_error")
        try:
            validator_func = _operator_precondition_validator(graph, node, operator)
            validator_func(graph, node)
            return True
        except Exception:
            if raise_on_error:
                raise
            return False

    def validate_graph_structure(
        self,
        graph: TNFRGraph,
        raise_on_error: bool = True,
    ) -> dict[str, Any]:
        """Run the graph owner's configured node and sigma checks.

        Performs structural validation including:
        - Node attribute completeness
        - EPI bounds and grid uniformity
        - Structural frequency ranges
        - Glyph provenance and the sigma norm check

        This does not collect the tetrad or certify coherence/persistence.

        Parameters
        ----------
        graph : TNFRGraph
            Graph to validate.
        raise_on_error : bool, optional
            Whether to raise exception on failure (default: True).

        Returns
        -------
        dict[str, Any]
            Validation results including passed checks and any errors.

        Raises
        ------
        TNFRValueError
            If structural validation fails and raise_on_error is True.
        """
        _require_boolean_flag(raise_on_error, "raise_on_error")
        if not self._enable_graph_validation:
            return {
                "passed": True,
                "skipped": True,
                "message": "Graph validation disabled",
            }

        from .graph import run_validators
        from .input_validation import validate_tnfr_graph

        try:
            validate_tnfr_graph(graph)
            run_validators(graph)
            return {"passed": True, "message": "Graph structure valid"}
        except Exception as e:
            if raise_on_error:
                raise
            return {"passed": False, "error": str(e)}

    def validate_runtime_canonical(
        self,
        graph: TNFRGraph,
        raise_on_error: bool = True,
    ) -> dict[str, Any]:
        """Validate runtime canonical constraints.

        Applies the runtime owner's configured clamps, refreshes maxima and
        checks graph contracts. This mutates the graph and can leave applied
        clamps even if validation fails. It is not a read-only or atomic check.

        Parameters
        ----------
        graph : TNFRGraph
            Graph to validate.
        raise_on_error : bool, optional
            Whether to raise exception on failure (default: True).

        Returns
        -------
        dict[str, Any]
            Validation results.

        Raises
        ------
        Exception
            If runtime validation fails and raise_on_error is True.
        """
        _require_boolean_flag(raise_on_error, "raise_on_error")
        if not self._enable_runtime_validation:
            return {
                "passed": True,
                "skipped": True,
                "message": "Runtime validation disabled",
            }

        from .input_validation import validate_tnfr_graph
        from .runtime import validate_canon

        try:
            validate_tnfr_graph(graph)
            outcome = validate_canon(graph)
            result = {
                "passed": outcome.passed,
                "summary": outcome.summary,
                "artifacts": outcome.artifacts,
            }
            if not outcome.passed:
                errors = outcome.summary.get("errors", ())
                result["error"] = (
                    "; ".join(map(str, errors))
                    if errors
                    else "Runtime canonical validation failed"
                )
                if raise_on_error:
                    raise TNFRValueError(result["error"])
            return result
        except Exception as e:
            if raise_on_error:
                raise
            return {"passed": False, "error": str(e)}

    def validate_graph(
        self,
        graph: TNFRGraph,
        severity_filter: InvariantSeverity | None = None,
        use_cache: bool = True,
        include_graph_validation: bool = True,
        include_runtime_validation: bool = False,
    ) -> list[InvariantViolation]:
        """Evaluate the configured graph and invariant checks on the live graph.

        The requested checks always execute afresh; graph identity does not
        authenticate their dependencies. Returned violation evidence is copied,
        preserving target node identities. Selected checks include:
        - Ten legacy invariant diagnostics and any supplied custom validators
        - Optional graph structure validation
        - Optional runtime canonical validation

        Parameters
        ----------
        graph : TNFRGraph
            Graph to validate against TNFR invariants.
        severity_filter : InvariantSeverity, optional
            Only return violations of this severity level.
        use_cache : bool, optional
            Compatibility argument; live graph results are never reused.
        include_graph_validation : bool, optional
            Include graph structure validation (default: True).
        include_runtime_validation : bool, optional
            Include the mutating runtime clamp/validation pass (default: False).

        Returns
        -------
        list[InvariantViolation]
            list of detected violations.

        Examples
        --------
        >>> validator = TNFRValidator()
        >>> violations = validator.validate_graph(graph)
        >>> if violations:
        ...     print(validator.generate_report(violations))
        """
        for name, value in (
            ("use_cache", use_cache),
            ("include_graph_validation", include_graph_validation),
            ("include_runtime_validation", include_runtime_validation),
        ):
            _require_boolean_flag(value, name)

        all_violations: list[InvariantViolation] = []

        # Run graph structure validation if enabled
        if include_graph_validation and self._enable_graph_validation:
            try:
                result = self.validate_graph_structure(graph, raise_on_error=False)
                if not result.get("passed", False):
                    all_violations.append(
                        InvariantViolation(
                            invariant_id=4,  # Operator closure
                            severity=InvariantSeverity.ERROR,
                            description=f"Graph structure validation failed: {result.get('error', 'Unknown error')}",
                            suggestion="Check graph structure and node attributes",
                        )
                    )
            except Exception as e:
                all_violations.append(
                    InvariantViolation(
                        invariant_id=4,
                        severity=InvariantSeverity.CRITICAL,
                        description=f"Graph structure validator failed: {str(e)}",
                        suggestion="Check graph structure validator implementation",
                    )
                )

        # Run runtime canonical validation if enabled
        if include_runtime_validation and self._enable_runtime_validation:
            try:
                result = self.validate_runtime_canonical(graph, raise_on_error=False)
                if not result.get("passed", False):
                    all_violations.append(
                        InvariantViolation(
                            invariant_id=8,  # Controlled determinism
                            severity=InvariantSeverity.WARNING,
                            description=f"Runtime canonical validation failed: {result.get('error', 'Unknown error')}",
                            suggestion="Check canonical clamps and runtime contracts",
                        )
                    )
            except Exception as e:
                all_violations.append(
                    InvariantViolation(
                        invariant_id=8,
                        severity=InvariantSeverity.WARNING,
                        description=f"Runtime validator failed: {str(e)}",
                        suggestion="Check runtime validator implementation",
                    )
                )

        # Run invariant validators
        for validator in self._invariant_validators + self._custom_validators:
            try:
                violations = validator.validate(graph)
                all_violations.extend(violations)
            except Exception as e:
                # If validator fails, it's a critical error
                all_violations.append(
                    InvariantViolation(
                        invariant_id=validator.invariant_id,
                        severity=InvariantSeverity.CRITICAL,
                        description=f"Validator execution failed: {str(e)}",
                        suggestion="Check validator implementation",
                    )
                )

        # Filter by severity if specified
        if severity_filter:
            all_violations = [
                v for v in all_violations if v.severity == severity_filter
            ]

        node_identities = {
            id(violation.node_id): violation.node_id
            for violation in all_violations
            if violation.node_id is not None
        }
        return deepcopy(all_violations, node_identities)

    def validate_and_raise(
        self,
        graph: TNFRGraph,
        min_severity: InvariantSeverity = InvariantSeverity.ERROR,
    ) -> None:
        """Validates and raises exception if violations of minimum severity are found.

        Parameters
        ----------
        graph : TNFRGraph
            Graph to validate.
        min_severity : InvariantSeverity
            Minimum severity level to trigger exception (default: ERROR).

        Raises
        ------
        TNFRValidationError
            If violations of minimum severity or higher are found.
        """
        violations = self.validate_graph(graph)

        # Filter violations by minimum severity
        severity_order = {
            InvariantSeverity.INFO: -1,
            InvariantSeverity.WARNING: 0,
            InvariantSeverity.ERROR: 1,
            InvariantSeverity.CRITICAL: 2,
        }

        critical_violations = [
            v
            for v in violations
            if severity_order[v.severity] >= severity_order[min_severity]
        ]

        if critical_violations:
            raise TNFRValidationError(critical_violations)

    def generate_report(self, violations: list[InvariantViolation]) -> str:
        """Genera reporte human-readable de violaciones.

        Parameters
        ----------
        violations : list[InvariantViolation]
            list of violations to report.

        Returns
        -------
        str
            Human-readable report.
        """
        if not violations:
            return "✅ No TNFR invariant violations found."

        report_lines = ["\n🚨 TNFR Invariant Violations Detected:\n"]

        # Group by severity
        by_severity: dict[InvariantSeverity, list[InvariantViolation]] = {}
        for v in violations:
            if v.severity not in by_severity:
                by_severity[v.severity] = []
            by_severity[v.severity].append(v)

        # Report by severity
        severity_icons = {
            InvariantSeverity.INFO: "ℹ️",
            InvariantSeverity.WARNING: "⚠️",
            InvariantSeverity.ERROR: "❌",
            InvariantSeverity.CRITICAL: "💥",
        }

        for severity in [
            InvariantSeverity.CRITICAL,
            InvariantSeverity.ERROR,
            InvariantSeverity.WARNING,
            InvariantSeverity.INFO,
        ]:
            if severity in by_severity:
                report_lines.append(
                    f"\n{severity_icons[severity]} {severity.value.upper()} "
                    f"({len(by_severity[severity])}):\n"
                )

                for violation in by_severity[severity]:
                    report_lines.append(
                        f"  Invariant #{violation.invariant_id}: {violation.description}"
                    )
                    if violation.node_id is not None:
                        report_lines.append(f"    Node: {violation.node_id}")
                    if violation.expected_value is not None:
                        report_lines.append(f"    Expected: {violation.expected_value}")
                    if violation.actual_value is not None:
                        report_lines.append(f"    Actual: {violation.actual_value}")
                    if violation.suggestion:
                        report_lines.append(
                            f"    💡 Suggestion: {violation.suggestion}"
                        )
                    report_lines.append("")

        return "\n".join(report_lines)

    def export_to_json(self, violations: list[InvariantViolation]) -> str:
        """Export violations to JSON format.

        Parameters
        ----------
        violations : list[InvariantViolation]
            list of violations to export.

        Returns
        -------
        str
            JSON-formatted string of violations.
        """
        import json

        violations_data = []
        for v in violations:
            violations_data.append(
                {
                    "invariant_id": v.invariant_id,
                    "severity": v.severity.value,
                    "description": v.description,
                    "node_id": v.node_id,
                    "expected_value": (
                        str(v.expected_value) if v.expected_value is not None else None
                    ),
                    "actual_value": (
                        str(v.actual_value) if v.actual_value is not None else None
                    ),
                    "suggestion": v.suggestion,
                }
            )

        return json.dumps(
            {
                "total_violations": len(violations),
                "by_severity": {
                    InvariantSeverity.CRITICAL.value: len(
                        [
                            v
                            for v in violations
                            if v.severity == InvariantSeverity.CRITICAL
                        ]
                    ),
                    InvariantSeverity.ERROR.value: len(
                        [v for v in violations if v.severity == InvariantSeverity.ERROR]
                    ),
                    InvariantSeverity.WARNING.value: len(
                        [
                            v
                            for v in violations
                            if v.severity == InvariantSeverity.WARNING
                        ]
                    ),
                    InvariantSeverity.INFO.value: len(
                        [v for v in violations if v.severity == InvariantSeverity.INFO]
                    ),
                },
                "violations": violations_data,
            },
            indent=2,
        )

    def export_to_html(self, violations: list[InvariantViolation]) -> str:
        """Export violations to HTML format.

        Parameters
        ----------
        violations : list[InvariantViolation]
            list of violations to export.

        Returns
        -------
        str
            HTML-formatted string of violations.
        """
        if not violations:
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>TNFR Validation Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .success { color: green; font-size: 24px; }
                </style>
            </head>
            <body>
                <h1>TNFR Validation Report</h1>
                <p class="success">✅ No TNFR invariant violations found.</p>
            </body>
            </html>
            """

        # Group by severity
        by_severity: dict[InvariantSeverity, list[InvariantViolation]] = {}
        for v in violations:
            if v.severity not in by_severity:
                by_severity[v.severity] = []
            by_severity[v.severity].append(v)

        severity_colors = {
            InvariantSeverity.INFO: "#17a2b8",
            InvariantSeverity.WARNING: "#ffc107",
            InvariantSeverity.ERROR: "#dc3545",
            InvariantSeverity.CRITICAL: "#6f42c1",
        }

        html_parts = [
            """
        <!DOCTYPE html>
        <html>
        <head>
            <title>TNFR Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                h1 {{ color: #333; }}
                .summary {{ background: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .severity-section {{ background: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                .severity-header {{ font-size: 20px; font-weight: bold; margin-bottom: 15px; }}
                .violation {{ background: #f9f9f9; padding: 15px; margin-bottom: 10px; border-left: 4px solid; border-radius: 3px; }}
                .violation-title {{ font-weight: bold; margin-bottom: 5px; }}
                .violation-detail {{ margin-left: 20px; color: #666; }}
                .suggestion {{ background: #e7f5ff; padding: 10px; margin-top: 10px; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <h1>🚨 TNFR Validation Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Total Violations:</strong> {}</p>
        """.format(
                len(violations)
            )
        ]

        for severity in [
            InvariantSeverity.CRITICAL,
            InvariantSeverity.ERROR,
            InvariantSeverity.WARNING,
            InvariantSeverity.INFO,
        ]:
            count = len(by_severity.get(severity, []))
            if count > 0:
                html_parts.append(
                    f"<p><strong>{severity.value.upper()}:</strong> {count}</p>"
                )

        html_parts.append("</div>")

        for severity in [
            InvariantSeverity.CRITICAL,
            InvariantSeverity.ERROR,
            InvariantSeverity.WARNING,
            InvariantSeverity.INFO,
        ]:
            if severity in by_severity:
                color = severity_colors[severity]
                html_parts.append(
                    f"""
                <div class="severity-section">
                    <div class="severity-header" style="color: {color};">
                        {severity.value.upper()} ({len(by_severity[severity])})
                    </div>
                """
                )

                for violation in by_severity[severity]:
                    html_parts.append(
                        f"""
                    <div class="violation" style="border-left-color: {color};">
                        <div class="violation-title">
                            Invariant #{violation.invariant_id}: {escape(str(violation.description))}
                        </div>
                    """
                    )

                    if violation.node_id is not None:
                        html_parts.append(
                            f'<div class="violation-detail"><strong>Node:</strong> {escape(str(violation.node_id))}</div>'
                        )

                    if violation.expected_value is not None:
                        html_parts.append(
                            f'<div class="violation-detail"><strong>Expected:</strong> {escape(str(violation.expected_value))}</div>'
                        )
                    if violation.actual_value is not None:
                        html_parts.append(
                            f'<div class="violation-detail"><strong>Actual:</strong> {escape(str(violation.actual_value))}</div>'
                        )

                    if violation.suggestion:
                        html_parts.append(
                            f'<div class="suggestion">💡 <strong>Suggestion:</strong> {escape(str(violation.suggestion))}</div>'
                        )

                    html_parts.append("</div>")

                html_parts.append("</div>")

        html_parts.append(
            """
        </body>
        </html>
        """
        )

        return "".join(html_parts)


class TNFRValidationError(TNFRValueError):
    """Exception raised when TNFR invariant violations are detected."""

    def __init__(self, violations: list[InvariantViolation]) -> None:
        self.violations = violations
        validator = TNFRValidator()
        self.report = validator.generate_report(violations)
        super().__init__(
            message=self.report,
            context={"violation_count": len(violations)},
            suggestion="Review the validation report and correct invariant violations.",
        )

    def export_to_json(self, violations: list[InvariantViolation]) -> str:
        """Export through the shared validator report implementation."""
        return TNFRValidator().export_to_json(violations)

    def export_to_html(self, violations: list[InvariantViolation]) -> str:
        """Export through the shared validator report implementation."""
        return TNFRValidator().export_to_html(violations)
