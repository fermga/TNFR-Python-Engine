"""Integrated TNFR Validator for graph-level invariant validation.

This module provides the TNFRValidator class which orchestrates the validation
of all TNFR invariants against graph structures. It integrates multiple
invariant validators and provides comprehensive reporting.
"""

from __future__ import annotations

from typing import Optional

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
from ..types import TNFRGraph

__all__ = [
    "TNFRValidator",
    "TNFRValidationError",
]


class TNFRValidator:
    """Validador integrado para grafos TNFR."""

    def __init__(
        self, phase_coupling_threshold: float | None = None
    ) -> None:
        """Initialize TNFR validator with invariant validators.

        Parameters
        ----------
        phase_coupling_threshold : float, optional
            Threshold for phase difference in coupled nodes (default: π/2).
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
        
        # Cache for validation results (graph_id -> violations)
        self._validation_cache: dict[int, list[InvariantViolation]] = {}
        self._cache_enabled = False

    def add_custom_validator(self, validator: TNFRInvariant) -> None:
        """Permite agregar validadores personalizados.

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

    def validate_graph(
        self,
        graph: TNFRGraph,
        severity_filter: Optional[InvariantSeverity] = None,
        use_cache: bool = True,
    ) -> list[InvariantViolation]:
        """Valida grafo contra todos los invariantes TNFR.

        Parameters
        ----------
        graph : TNFRGraph
            Graph to validate against TNFR invariants.
        severity_filter : InvariantSeverity, optional
            Only return violations of this severity level.
        use_cache : bool, optional
            Whether to use cached results if available (default: True).

        Returns
        -------
        list[InvariantViolation]
            List of detected violations.
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

        # Ejecutar validadores de invariantes
        for validator in self._invariant_validators + self._custom_validators:
            try:
                violations = validator.validate(graph)
                all_violations.extend(violations)
            except Exception as e:
                # Si el validador falla, es un error crítico
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
            all_violations = [
                v for v in all_violations if v.severity == severity_filter
            ]

        return all_violations

    def validate_and_raise(
        self,
        graph: TNFRGraph,
        min_severity: InvariantSeverity = InvariantSeverity.ERROR,
    ) -> None:
        """Valida y lanza excepción si encuentra violaciones de severidad mínima.

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

        # Filtrar violaciones por severidad mínima
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
                        report_lines.append(
                            f"    Expected: {violation.expected_value}"
                        )
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
            List of violations to export.
        
        Returns
        -------
        str
            JSON-formatted string of violations.
        """
        import json
        
        violations_data = []
        for v in violations:
            violations_data.append({
                "invariant_id": v.invariant_id,
                "severity": v.severity.value,
                "description": v.description,
                "node_id": v.node_id,
                "expected_value": str(v.expected_value) if v.expected_value else None,
                "actual_value": str(v.actual_value) if v.actual_value else None,
                "suggestion": v.suggestion,
            })
        
        return json.dumps({
            "total_violations": len(violations),
            "by_severity": {
                InvariantSeverity.CRITICAL.value: len([v for v in violations if v.severity == InvariantSeverity.CRITICAL]),
                InvariantSeverity.ERROR.value: len([v for v in violations if v.severity == InvariantSeverity.ERROR]),
                InvariantSeverity.WARNING.value: len([v for v in violations if v.severity == InvariantSeverity.WARNING]),
                InvariantSeverity.INFO.value: len([v for v in violations if v.severity == InvariantSeverity.INFO]),
            },
            "violations": violations_data
        }, indent=2)
    
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
        
        html_parts = ["""
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
                <p><strong>Total Violations:</strong> {{}}</p>
        """.format(len(violations))]
        
        for severity in [InvariantSeverity.CRITICAL, InvariantSeverity.ERROR, InvariantSeverity.WARNING, InvariantSeverity.INFO]:
            count = len(by_severity.get(severity, []))
            if count > 0:
                html_parts.append(f'<p><strong>{severity.value.upper()}:</strong> {count}</p>')
        
        html_parts.append("</div>")
        
        for severity in [InvariantSeverity.CRITICAL, InvariantSeverity.ERROR, InvariantSeverity.WARNING, InvariantSeverity.INFO]:
            if severity in by_severity:
                color = severity_colors[severity]
                html_parts.append(f"""
                <div class="severity-section">
                    <div class="severity-header" style="color: {color};">
                        {severity.value.upper()} ({len(by_severity[severity])})
                    </div>
                """)
                
                for violation in by_severity[severity]:
                    html_parts.append(f"""
                    <div class="violation" style="border-left-color: {color};">
                        <div class="violation-title">
                            Invariant #{violation.invariant_id}: {violation.description}
                        </div>
                    """)
                    
                    if violation.node_id:
                        html_parts.append(f'<div class="violation-detail"><strong>Node:</strong> {violation.node_id}</div>')
                    
                    if violation.expected_value and violation.actual_value:
                        html_parts.append(f'<div class="violation-detail"><strong>Expected:</strong> {violation.expected_value}</div>')
                        html_parts.append(f'<div class="violation-detail"><strong>Actual:</strong> {violation.actual_value}</div>')
                    
                    if violation.suggestion:
                        html_parts.append(f'<div class="suggestion">💡 <strong>Suggestion:</strong> {violation.suggestion}</div>')
                    
                    html_parts.append("</div>")
                
                html_parts.append("</div>")
        
        html_parts.append("""
        </body>
        </html>
        """)
        
        return "".join(html_parts)


class TNFRValidationError(Exception):
    """Excepción lanzada cuando se detectan violaciones de invariantes TNFR."""

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
            violations_data.append({
                "invariant_id": v.invariant_id,
                "severity": v.severity.value,
                "description": v.description,
                "node_id": v.node_id,
                "expected_value": str(v.expected_value) if v.expected_value else None,
                "actual_value": str(v.actual_value) if v.actual_value else None,
                "suggestion": v.suggestion,
            })
        
        return json.dumps({
            "total_violations": len(violations),
            "by_severity": {
                InvariantSeverity.CRITICAL.value: len([v for v in violations if v.severity == InvariantSeverity.CRITICAL]),
                InvariantSeverity.ERROR.value: len([v for v in violations if v.severity == InvariantSeverity.ERROR]),
                InvariantSeverity.WARNING.value: len([v for v in violations if v.severity == InvariantSeverity.WARNING]),
                InvariantSeverity.INFO.value: len([v for v in violations if v.severity == InvariantSeverity.INFO]),
            },
            "violations": violations_data
        }, indent=2)
    
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
        
        html_parts = ["""
        <!DOCTYPE html>
        <html>
        <head>
            <title>TNFR Validation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
                h1 { color: #333; }
                .summary { background: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
                .severity-section { background: white; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
                .severity-header { font-size: 20px; font-weight: bold; margin-bottom: 15px; }
                .violation { background: #f9f9f9; padding: 15px; margin-bottom: 10px; border-left: 4px solid; border-radius: 3px; }
                .violation-title { font-weight: bold; margin-bottom: 5px; }
                .violation-detail { margin-left: 20px; color: #666; }
                .suggestion { background: #e7f5ff; padding: 10px; margin-top: 10px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>🚨 TNFR Validation Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Total Violations:</strong> {}</p>
        """.format(len(violations))]
        
        for severity in [InvariantSeverity.CRITICAL, InvariantSeverity.ERROR, InvariantSeverity.WARNING, InvariantSeverity.INFO]:
            count = len(by_severity.get(severity, []))
            if count > 0:
                html_parts.append(f'<p><strong>{severity.value.upper()}:</strong> {count}</p>')
        
        html_parts.append("</div>")
        
        for severity in [InvariantSeverity.CRITICAL, InvariantSeverity.ERROR, InvariantSeverity.WARNING, InvariantSeverity.INFO]:
            if severity in by_severity:
                color = severity_colors[severity]
                html_parts.append(f"""
                <div class="severity-section">
                    <div class="severity-header" style="color: {color};">
                        {severity.value.upper()} ({len(by_severity[severity])})
                    </div>
                """)
                
                for violation in by_severity[severity]:
                    html_parts.append(f"""
                    <div class="violation" style="border-left-color: {color};">
                        <div class="violation-title">
                            Invariant #{violation.invariant_id}: {violation.description}
                        </div>
                    """)
                    
                    if violation.node_id:
                        html_parts.append(f'<div class="violation-detail"><strong>Node:</strong> {violation.node_id}</div>')
                    
                    if violation.expected_value and violation.actual_value:
                        html_parts.append(f'<div class="violation-detail"><strong>Expected:</strong> {violation.expected_value}</div>')
                        html_parts.append(f'<div class="violation-detail"><strong>Actual:</strong> {violation.actual_value}</div>')
                    
                    if violation.suggestion:
                        html_parts.append(f'<div class="suggestion">💡 <strong>Suggestion:</strong> {violation.suggestion}</div>')
                    
                    html_parts.append("</div>")
                
                html_parts.append("</div>")
        
        html_parts.append("""
        </body>
        </html>
        """)
        
        return "".join(html_parts)
