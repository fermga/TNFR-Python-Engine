"""TNFR Invariant Validators.

This module implements the 10 canonical TNFR invariants as described in AGENTS.md.
Each invariant is a structural constraint that must be preserved to maintain
coherence within the TNFR paradigm.

Canonical Invariants:
1. EPI as coherent form: changes only via structural operators
2. Structural units: νf expressed in Hz_str (structural hertz)
3. ΔNFR semantics: sign and magnitude modulate reorganization rate
4. Operator closure: composition yields valid TNFR states
5. Phase check: explicit phase verification for coupling
6. Node birth/collapse: minimal conditions maintained
7. Operational fractality: EPIs can nest without losing identity
8. Controlled determinism: reproducible and traceable
9. Structural metrics: expose C(t), Si, phase, νf
10. Domain neutrality: trans-scale and trans-domain
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from ..constants import DEFAULTS, DNFR_PRIMARY, EPI_PRIMARY, THETA_PRIMARY, VF_PRIMARY
from ..types import TNFRGraph

__all__ = [
    "InvariantSeverity",
    "InvariantViolation",
    "TNFRInvariant",
    "Invariant1_EPIOnlyThroughOperators",
    "Invariant2_VfInHzStr",
    "Invariant5_ExplicitPhaseChecks",
]


class InvariantSeverity(Enum):
    """Severity levels for invariant violations."""

    WARNING = "warning"  # Inconsistencia menor
    ERROR = "error"  # Violación que impide ejecución
    CRITICAL = "critical"  # Corrupción de datos


@dataclass
class InvariantViolation:
    """Descripción detallada de violación de invariante."""

    invariant_id: int
    severity: InvariantSeverity
    description: str
    node_id: Optional[str] = None
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    suggestion: Optional[str] = None


class TNFRInvariant(ABC):
    """Base class para validadores de invariantes TNFR."""

    @property
    @abstractmethod
    def invariant_id(self) -> int:
        """Número de invariante TNFR (1-10)."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Descripción human-readable del invariante."""

    @abstractmethod
    def validate(self, graph: TNFRGraph) -> list[InvariantViolation]:
        """Valida invariante en el grafo, retorna violaciones encontradas."""


class Invariant1_EPIOnlyThroughOperators(TNFRInvariant):
    """Invariante 1: EPI cambia solo a través de operadores estructurales."""

    invariant_id = 1
    description = "EPI changes only through structural operators"

    def __init__(self) -> None:
        self._previous_epi_values: dict[Any, float] = {}

    def validate(self, graph: TNFRGraph) -> list[InvariantViolation]:
        violations = []

        # Get configuration bounds
        config = getattr(graph, "graph", {})
        epi_min = config.get("EPI_MIN", DEFAULTS.get("EPI_MIN", 0.0))
        epi_max = config.get("EPI_MAX", DEFAULTS.get("EPI_MAX", 1.0))

        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            current_epi = node_data.get(EPI_PRIMARY, 0.0)

            # Handle complex EPI structures (dict, complex numbers)
            # Extract scalar value for validation
            if isinstance(current_epi, dict):
                # EPI can be a dict with 'continuous', 'discrete', 'grid' keys
                # Try to extract a scalar value for validation
                if "continuous" in current_epi:
                    epi_value = current_epi["continuous"]
                    if isinstance(epi_value, (tuple, list)) and len(epi_value) > 0:
                        epi_value = epi_value[0]
                    if isinstance(epi_value, complex):
                        epi_value = abs(epi_value)
                    current_epi = float(epi_value) if isinstance(epi_value, (int, float, complex)) else 0.0
                else:
                    # Skip validation for complex structures we can't interpret
                    continue

            elif isinstance(current_epi, complex):
                # For complex numbers, use magnitude
                current_epi = abs(current_epi)

            # Verificar rango válido de EPI
            if not (epi_min <= current_epi <= epi_max):
                violations.append(
                    InvariantViolation(
                        invariant_id=1,
                        severity=InvariantSeverity.ERROR,
                        description=f"EPI out of valid range [{epi_min},{epi_max}]",
                        node_id=str(node_id),
                        expected_value=f"{epi_min} <= EPI <= {epi_max}",
                        actual_value=current_epi,
                        suggestion="Check operator implementation for EPI clamping",
                    )
                )

            # Verificar que EPI es un número finito
            if not isinstance(current_epi, (int, float)) or not math.isfinite(
                current_epi
            ):
                violations.append(
                    InvariantViolation(
                        invariant_id=1,
                        severity=InvariantSeverity.CRITICAL,
                        description="EPI is not a finite number",
                        node_id=str(node_id),
                        expected_value="finite float",
                        actual_value=f"{type(current_epi).__name__}: {current_epi}",
                        suggestion="Check operator implementation for EPI assignment",
                    )
                )

            # Detectar cambios no autorizados (requiere tracking)
            # Solo verificar si hay un operador previo registrado
            if hasattr(graph, "_last_operator_applied"):
                if node_id in self._previous_epi_values:
                    prev_epi = self._previous_epi_values[node_id]
                    if abs(current_epi - prev_epi) > 1e-10:  # Cambio detectado
                        if not graph._last_operator_applied:
                            violations.append(
                                InvariantViolation(
                                    invariant_id=1,
                                    severity=InvariantSeverity.CRITICAL,
                                    description="EPI changed without operator application",
                                    node_id=str(node_id),
                                    expected_value=prev_epi,
                                    actual_value=current_epi,
                                    suggestion="Ensure all EPI modifications go through structural operators",
                                )
                            )

        # Actualizar tracking
        for node_id in graph.nodes():
            epi_value = graph.nodes[node_id].get(EPI_PRIMARY, 0.0)
            # Store scalar value for tracking
            if isinstance(epi_value, dict) and "continuous" in epi_value:
                epi_val = epi_value["continuous"]
                if isinstance(epi_val, (tuple, list)) and len(epi_val) > 0:
                    epi_val = epi_val[0]
                if isinstance(epi_val, complex):
                    epi_val = abs(epi_val)
                epi_value = float(epi_val) if isinstance(epi_val, (int, float, complex)) else 0.0
            elif isinstance(epi_value, complex):
                epi_value = abs(epi_value)
            
            self._previous_epi_values[node_id] = epi_value

        return violations


class Invariant2_VfInHzStr(TNFRInvariant):
    """Invariante 2: νf stays in Hz_str units."""

    invariant_id = 2
    description = "νf stays in Hz_str units"

    def validate(self, graph: TNFRGraph) -> list[InvariantViolation]:
        violations = []

        # Get configuration bounds
        config = getattr(graph, "graph", {})
        vf_min = config.get("VF_MIN", DEFAULTS.get("VF_MIN", 0.001))
        vf_max = config.get("VF_MAX", DEFAULTS.get("VF_MAX", 1000.0))

        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            vf = node_data.get(VF_PRIMARY, 0.0)

            # Verificar rango estructural válido (Hz_str)
            if not (vf_min <= vf <= vf_max):
                violations.append(
                    InvariantViolation(
                        invariant_id=2,
                        severity=InvariantSeverity.ERROR,
                        description=f"νf outside typical Hz_str range [{vf_min}, {vf_max}]",
                        node_id=str(node_id),
                        expected_value=f"{vf_min} <= νf <= {vf_max} Hz_str",
                        actual_value=vf,
                        suggestion="Verify νf units and operator calculations",
                    )
                )

            # Verificar que sea un número válido
            if not isinstance(vf, (int, float)) or not math.isfinite(vf):
                violations.append(
                    InvariantViolation(
                        invariant_id=2,
                        severity=InvariantSeverity.CRITICAL,
                        description="νf is not a finite number",
                        node_id=str(node_id),
                        expected_value="finite float",
                        actual_value=f"{type(vf).__name__}: {vf}",
                        suggestion="Check operator implementation for νf assignment",
                    )
                )

            # Verificar que νf sea positivo (requerimiento estructural)
            if isinstance(vf, (int, float)) and vf <= 0:
                violations.append(
                    InvariantViolation(
                        invariant_id=2,
                        severity=InvariantSeverity.ERROR,
                        description="νf must be positive (structural frequency)",
                        node_id=str(node_id),
                        expected_value="νf > 0",
                        actual_value=vf,
                        suggestion="Structural frequency must be positive for coherent nodes",
                    )
                )

        return violations


class Invariant5_ExplicitPhaseChecks(TNFRInvariant):
    """Invariante 5: Explicit phase checks for coupling."""

    invariant_id = 5
    description = "Explicit phase checks for coupling"

    def __init__(self, phase_coupling_threshold: float = math.pi / 2) -> None:
        self.phase_coupling_threshold = phase_coupling_threshold

    def validate(self, graph: TNFRGraph) -> list[InvariantViolation]:
        violations = []

        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            phase = node_data.get(THETA_PRIMARY, 0.0)

            # Verificar que phase sea un número finito
            if not isinstance(phase, (int, float)) or not math.isfinite(phase):
                violations.append(
                    InvariantViolation(
                        invariant_id=5,
                        severity=InvariantSeverity.CRITICAL,
                        description="Phase is not a finite number",
                        node_id=str(node_id),
                        expected_value="finite float",
                        actual_value=f"{type(phase).__name__}: {phase}",
                        suggestion="Check operator implementation for phase assignment",
                    )
                )
                continue

            # Verificar rango de fase [0, 2π] o normalizable
            # TNFR permite fases fuera de este rango si se pueden normalizar
            # Emitir warning si la fase no está en el rango canónico
            if not (0.0 <= phase <= 2 * math.pi):
                violations.append(
                    InvariantViolation(
                        invariant_id=5,
                        severity=InvariantSeverity.WARNING,
                        description="Phase outside [0, 2π] range (normalization possible)",
                        node_id=str(node_id),
                        expected_value="0.0 <= phase <= 2π",
                        actual_value=phase,
                        suggestion="Consider normalizing phase to [0, 2π] range",
                    )
                )

        # Verificar sincronización en nodos acoplados (edges)
        if hasattr(graph, "edges"):
            for edge in graph.edges():
                node1, node2 = edge
                phase1 = graph.nodes[node1].get(THETA_PRIMARY, 0.0)
                phase2 = graph.nodes[node2].get(THETA_PRIMARY, 0.0)

                # Verificar que ambas fases sean números finitos antes de calcular diferencia
                if not (
                    isinstance(phase1, (int, float))
                    and math.isfinite(phase1)
                    and isinstance(phase2, (int, float))
                    and math.isfinite(phase2)
                ):
                    continue

                phase_diff = abs(phase1 - phase2)
                # Considerar periodicidad
                phase_diff = min(phase_diff, 2 * math.pi - phase_diff)

                # Si la diferencia es muy grande, puede indicar desacoplamiento
                if phase_diff > self.phase_coupling_threshold:
                    violations.append(
                        InvariantViolation(
                            invariant_id=5,
                            severity=InvariantSeverity.WARNING,
                            description="Large phase difference between coupled nodes",
                            node_id=f"{node1}-{node2}",
                            expected_value=f"< {self.phase_coupling_threshold}",
                            actual_value=phase_diff,
                            suggestion="Check coupling strength or phase coordination",
                        )
                    )

        return violations
