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

    def add_custom_validator(self, validator: TNFRInvariant) -> None:
        """Permite agregar validadores personalizados.

        Parameters
        ----------
        validator : TNFRInvariant
            Custom validator implementing TNFRInvariant interface.
        """
        self._custom_validators.append(validator)

    def validate_graph(
        self,
        graph: TNFRGraph,
        severity_filter: Optional[InvariantSeverity] = None,
    ) -> list[InvariantViolation]:
        """Valida grafo contra todos los invariantes TNFR.

        Parameters
        ----------
        graph : TNFRGraph
            Graph to validate against TNFR invariants.
        severity_filter : InvariantSeverity, optional
            Only return violations of this severity level.

        Returns
        -------
        list[InvariantViolation]
            List of detected violations.
        """
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
            InvariantSeverity.WARNING: "⚠️",
            InvariantSeverity.ERROR: "❌",
            InvariantSeverity.CRITICAL: "💥",
        }

        for severity in [
            InvariantSeverity.CRITICAL,
            InvariantSeverity.ERROR,
            InvariantSeverity.WARNING,
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


class TNFRValidationError(Exception):
    """Excepción lanzada cuando se detectan violaciones de invariantes TNFR."""

    def __init__(self, violations: list[InvariantViolation]) -> None:
        self.violations = violations
        validator = TNFRValidator()
        self.report = validator.generate_report(violations)
        super().__init__(self.report)
