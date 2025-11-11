"""Unified TNFR Validation Pipeline.

This module provides the TNFRValidator class which serves as the canonical
entry point for all TNFR validation operations. It integrates:
- Invariant validation (10 canonical TNFR invariants)
- Input validation (parameters, types, bounds)
- Graph validation (structure, coherence)
- Runtime validation (canonical clamps, contracts)
- Security validation (injection prevention, type safety)
- Operator precondition validation
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .invariants import (
    InvariantSeverity,
    InvariantViolation,
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
    TNFRInvariant,
)
from ..types import NodeId, TNFRGraph

__all__ = [
    "TNFRValidator",
    "TNFRValidationError",
]


class TNFRValidator:
    """Unified TNFR Validation Pipeline.

    This class serves as the single entry point for all TNFR validation operations,
    consolidating scattered validation logic into a coherent pipeline that enforces
    all canonical TNFR invariants.

    Features
    --------
    - Validates 10 canonical TNFR invariants
    - Input validation with security checks
    - Graph structure and coherence validation
    - Runtime canonical validation
    - Operator precondition checking
    - Comprehensive reporting (text, JSON, HTML)
    - Optional result caching for performance

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

        # Cache for validation results (graph_id -> violations)
        self._validation_cache: dict[int, list[InvariantViolation]] = {}
        self._cache_enabled = False

    def add_custom_validator(self, validator: TNFRInvariant) -> None:
        """Add custom invariant validator.

        Parameters
        ----------
        validator : TNFRInvariant
            Custom validator implementing TNFRInvariant interface.
        """
        self._custom_validators.append(validator)

    def enable_cache(self, enabled: bool = True) -> None:
        """Enable or disable validation result caching.

        Parameters
        ----------
        enabled : bool
            Whether to enable caching (default: True).
        """
        self._cache_enabled = enabled
        if not enabled:
            self._validation_cache.clear()

    def clear_cache(self) -> None:
        """Clear the validation result cache."""
        self._validation_cache.clear()

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
        """Comprehensive unified validation pipeline (single entry point).

        This method provides a single entry point for all TNFR validation needs,
        consolidating input validation, graph validation, invariant checking,
        and operator preconditions into one call.

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
            Include runtime canonical validation (default: False).
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
        results: dict[str, Any] = {
            "passed": True,
            "inputs": {},
            "graph_structure": None,
            "runtime": None,
            "invariants": [],
            "operator_preconditions": None,
            "errors": [],
        }

        config = graph.graph if graph is not None else None

        # Input validation
        if epi is not None or vf is not None or theta is not None or dnfr is not None:
            try:
                results["inputs"] = self.validate_inputs(
                    epi=epi,
                    vf=vf,
                    theta=theta,
                    dnfr=dnfr,
                    node_id=node_id,
                    config=config,
                    raise_on_error=raise_on_error,
                )
                if "error" in results["inputs"]:
                    results["passed"] = False
                    results["errors"].append(f"Input validation: {results['inputs']['error']}")
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
                    results["errors"].append(f"Graph structure validation failed: {str(e)}")
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
                            if v.severity in (InvariantSeverity.ERROR, InvariantSeverity.CRITICAL)
                        ]
                        if critical_violations:
                            results["passed"] = False
                            results["errors"].append(
                                f"{len(critical_violations)} critical invariant violations found"
                            )
                except Exception as e:
                    results["passed"] = False
                    results["errors"].append(f"Invariant validation failed: {str(e)}")
                    if raise_on_error:
                        raise

            # Operator preconditions validation
            if operator is not None and node_id is not None:
                try:
                    results["operator_preconditions"] = self.validate_operator_preconditions(
                        graph,
                        node_id,
                        operator,
                        raise_on_error=raise_on_error,
                    )
                    if not results["operator_preconditions"]:
                        results["passed"] = False
                        results["errors"].append(
                            f"Operator '{operator}' preconditions not met for node {node_id}"
                        )
                except Exception as e:
                    results["passed"] = False
                    results["errors"].append(f"Operator precondition validation failed: {str(e)}")
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

        This method consolidates input validation for all TNFR structural parameters,
        enforcing type safety, bounds checking, and security constraints.

        Parameters
        ----------
        epi : Any, optional
            EPI (Primary Information Structure) value to validate.
        vf : Any, optional
            νf (structural frequency) value to validate.
        theta : Any, optional
            θ (phase) value to validate.
        dnfr : Any, optional
            ΔNFR (reorganization operator) value to validate.
        node_id : Any, optional
            Node identifier to validate.
        glyph : Any, optional
            Glyph enumeration to validate.
        graph : Any, optional
            TNFRGraph to validate.
        config : Mapping[str, Any], optional
            Configuration for bounds checking.
        raise_on_error : bool, optional
            Whether to raise exception on validation failure (default: True).

        Returns
        -------
        dict[str, Any]
            Dictionary with validation results for each parameter.
            Keys: parameter names, Values: validation status or validated values.

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
        if not self._enable_input_validation:
            return {}

        from .input_validation import (
            validate_epi_value,
            validate_vf_value,
            validate_theta_value,
            validate_dnfr_value,
            validate_node_id,
            validate_glyph,
            validate_tnfr_graph,
        )

        results = {}

        try:
            if epi is not None:
                results["epi"] = validate_epi_value(epi, config=config)

            if vf is not None:
                results["vf"] = validate_vf_value(vf, config=config)

            if theta is not None:
                results["theta"] = validate_theta_value(theta)

            if dnfr is not None:
                results["dnfr"] = validate_dnfr_value(dnfr, config=config)

            if node_id is not None:
                results["node_id"] = validate_node_id(node_id)

            if glyph is not None:
                results["glyph"] = validate_glyph(glyph)

            if graph is not None:
                results["graph"] = validate_tnfr_graph(graph)

        except Exception as e:
            if raise_on_error:
                raise
            results["error"] = str(e)

        return results

    def validate_operator_preconditions(
        self,
        graph: TNFRGraph,
        node: NodeId,
        operator: str,
        raise_on_error: bool = True,
    ) -> bool:
        """Validate operator preconditions before application.

        Each TNFR structural operator has specific requirements that must be met
        before execution to maintain structural invariants.

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
        from ..operators import preconditions

        validator_map = {
            "emission": preconditions.validate_emission,
            "reception": preconditions.validate_reception,
            "coherence": preconditions.validate_coherence,
            "dissonance": preconditions.validate_dissonance,
            "coupling": preconditions.validate_coupling,
            "resonance": preconditions.validate_resonance,
            "silence": preconditions.validate_silence,
            "expansion": preconditions.validate_expansion,
            "contraction": preconditions.validate_contraction,
            "self_organization": preconditions.validate_self_organization,
            "mutation": preconditions.validate_mutation,
            "transition": preconditions.validate_transition,
            "recursivity": preconditions.validate_recursivity,
        }

        validator_func = validator_map.get(operator.lower())
        if validator_func is None:
            if raise_on_error:
                raise ValueError(f"Unknown operator: {operator}")
            return False

        try:
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
        """Validate graph structure and coherence.

        Performs structural validation including:
        - Node attribute completeness
        - EPI bounds and grid uniformity
        - Structural frequency ranges
        - Coherence metrics

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
        ValueError
            If structural validation fails and raise_on_error is True.
        """
        if not self._enable_graph_validation:
            return {"passed": True, "message": "Graph validation disabled"}

        from .graph import run_validators

        try:
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

        Applies canonical clamps and validates graph contracts at runtime.

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
        if not self._enable_runtime_validation:
            return {"passed": True, "message": "Runtime validation disabled"}

        from .runtime import validate_canon

        try:
            outcome = validate_canon(graph)
            return {
                "passed": outcome.passed,
                "summary": outcome.summary,
                "artifacts": outcome.artifacts,
            }
        except Exception as e:
            if raise_on_error:
                raise
            return {"passed": False, "error": str(e)}

    def validate_graph(
        self,
        graph: TNFRGraph,
        severity_filter: Optional[InvariantSeverity] = None,
        use_cache: bool = True,
        include_graph_validation: bool = True,
        include_runtime_validation: bool = False,
    ) -> list[InvariantViolation]:
        """Validate graph against all TNFR invariants (unified pipeline).

        This is the main entry point for comprehensive graph validation,
        integrating all validation layers:
        - Invariant validation (10 canonical TNFR invariants)
        - Optional graph structure validation
        - Optional runtime canonical validation

        Parameters
        ----------
        graph : TNFRGraph
            Graph to validate against TNFR invariants.
        severity_filter : InvariantSeverity, optional
            Only return violations of this severity level.
        use_cache : bool, optional
            Whether to use cached results if available (default: True).
        include_graph_validation : bool, optional
            Include graph structure validation (default: True).
        include_runtime_validation : bool, optional
            Include runtime canonical validation (default: False).

        Returns
        -------
        list[InvariantViolation]
            List of detected violations.

        Examples
        --------
        >>> validator = TNFRValidator()
        >>> violations = validator.validate_graph(graph)
        >>> if violations:
        ...     print(validator.generate_report(violations))
        """
        # Check cache if enabled
        if self._cache_enabled and use_cache:
            graph_id = id(graph)
            if graph_id in self._validation_cache:
                all_violations = self._validation_cache[graph_id]
                # Apply severity filter if specified
                if severity_filter:
                    return [v for v in all_violations if v.severity == severity_filter]
                return all_violations

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

        # Cache results if enabled
        if self._cache_enabled:
            graph_id = id(graph)
            self._validation_cache[graph_id] = all_violations.copy()

        # Filtrar por severidad si se especifica
        if severity_filter:
            all_violations = [v for v in all_violations if v.severity == severity_filter]

        return all_violations

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
            v for v in violations if severity_order[v.severity] >= severity_order[min_severity]
        ]

        if critical_violations:
            raise TNFRValidationError(critical_violations)

    def generate_report(self, violations: list[InvariantViolation]) -> str:
        """Genera reporte human-readable de violaciones.

        Parameters
        ----------
        violations : list[InvariantViolation]
            List of violations to report.

        Returns
        -------
        str
            Human-readable report.
        """
        if not violations:
            return "✅ No TNFR invariant violations found."

        report_lines = ["\n🚨 TNFR Invariant Violations Detected:\n"]

        # Agrupar por severidad
        by_severity: dict[InvariantSeverity, list[InvariantViolation]] = {}
        for v in violations:
            if v.severity not in by_severity:
                by_severity[v.severity] = []
            by_severity[v.severity].append(v)

        # Reporte por severidad
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
                    if violation.node_id:
                        report_lines.append(f"    Node: {violation.node_id}")
                    if violation.expected_value and violation.actual_value:
                        report_lines.append(f"    Expected: {violation.expected_value}")
                        report_lines.append(f"    Actual: {violation.actual_value}")
                    if violation.suggestion:
                        report_lines.append(f"    💡 Suggestion: {violation.suggestion}")
                    report_lines.append("")

        return "\n".join(report_lines)

    def export_to_json(self, violations: list[InvariantViolation]) -> str:
        """Export violations to JSON format.

        Parameters
        ----------
        violations : list[InvariantViolation]
            List of violations to export.

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
                    "expected_value": (str(v.expected_value) if v.expected_value else None),
                    "actual_value": str(v.actual_value) if v.actual_value else None,
                    "suggestion": v.suggestion,
                }
            )

        return json.dumps(
            {
                "total_violations": len(violations),
                "by_severity": {
                    InvariantSeverity.CRITICAL.value: len(
                        [v for v in violations if v.severity == InvariantSeverity.CRITICAL]
                    ),
                    InvariantSeverity.ERROR.value: len(
                        [v for v in violations if v.severity == InvariantSeverity.ERROR]
                    ),
                    InvariantSeverity.WARNING.value: len(
                        [v for v in violations if v.severity == InvariantSeverity.WARNING]
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
            List of violations to export.

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
                html_parts.append(f"<p><strong>{severity.value.upper()}:</strong> {count}</p>")

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
                            Invariant #{violation.invariant_id}: {violation.description}
                        </div>
                    """
                    )

                    if violation.node_id:
                        html_parts.append(
                            f'<div class="violation-detail"><strong>Node:</strong> {violation.node_id}</div>'
                        )

                    if violation.expected_value and violation.actual_value:
                        html_parts.append(
                            f'<div class="violation-detail"><strong>Expected:</strong> {violation.expected_value}</div>'
                        )
                        html_parts.append(
                            f'<div class="violation-detail"><strong>Actual:</strong> {violation.actual_value}</div>'
                        )

                    if violation.suggestion:
                        html_parts.append(
                            f'<div class="suggestion">💡 <strong>Suggestion:</strong> {violation.suggestion}</div>'
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


class TNFRValidationError(Exception):
    """Exception raised when TNFR invariant violations are detected."""

    def __init__(self, violations: list[InvariantViolation]) -> None:
        self.violations = violations
        validator = TNFRValidator()
        self.report = validator.generate_report(violations)
        super().__init__(self.report)

    def export_to_json(self, violations: list[InvariantViolation]) -> str:
        """Export violations to JSON format.

        Parameters
        ----------
        violations : list[InvariantViolation]
            List of violations to export.

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
                    "expected_value": (str(v.expected_value) if v.expected_value else None),
                    "actual_value": str(v.actual_value) if v.actual_value else None,
                    "suggestion": v.suggestion,
                }
            )

        return json.dumps(
            {
                "total_violations": len(violations),
                "by_severity": {
                    InvariantSeverity.CRITICAL.value: len(
                        [v for v in violations if v.severity == InvariantSeverity.CRITICAL]
                    ),
                    InvariantSeverity.ERROR.value: len(
                        [v for v in violations if v.severity == InvariantSeverity.ERROR]
                    ),
                    InvariantSeverity.WARNING.value: len(
                        [v for v in violations if v.severity == InvariantSeverity.WARNING]
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
            List of violations to export.

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
                html_parts.append(f"<p><strong>{severity.value.upper()}:</strong> {count}</p>")

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
                            Invariant #{violation.invariant_id}: {violation.description}
                        </div>
                    """
                    )

                    if violation.node_id:
                        html_parts.append(
                            f'<div class="violation-detail"><strong>Node:</strong> {violation.node_id}</div>'
                        )

                    if violation.expected_value and violation.actual_value:
                        html_parts.append(
                            f'<div class="violation-detail"><strong>Expected:</strong> {violation.expected_value}</div>'
                        )
                        html_parts.append(
                            f'<div class="violation-detail"><strong>Actual:</strong> {violation.actual_value}</div>'
                        )

                    if violation.suggestion:
                        html_parts.append(
                            f'<div class="suggestion">💡 <strong>Suggestion:</strong> {violation.suggestion}</div>'
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
