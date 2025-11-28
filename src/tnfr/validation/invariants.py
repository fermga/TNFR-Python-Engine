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
    "Invariant3_DNFRSemantics",
    "Invariant4_OperatorClosure",
    "Invariant5_ExplicitPhaseChecks",
    "Invariant6_NodeBirthCollapse",
    "Invariant7_OperationalFractality",
    "Invariant8_ControlledDeterminism",
    "Invariant9_StructuralMetrics",
    "Invariant10_DomainNeutrality",
]


class InvariantSeverity(Enum):
    """Severity levels for invariant violations."""

    INFO = "info"  # Information, not a problem
    WARNING = "warning"  # Minor inconsistency
    ERROR = "error"  # Violation that prevents execution
    CRITICAL = "critical"  # Data corruption


@dataclass
class InvariantViolation:
    """Detailed description of invariant violation."""

    invariant_id: int
    severity: InvariantSeverity
    description: str
    node_id: Optional[str] = None
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    suggestion: Optional[str] = None


class TNFRInvariant(ABC):
    """Base class for TNFR invariant validators."""

    @property
    @abstractmethod
    def invariant_id(self) -> int:
        """TNFR invariant number (1-10)."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the invariant."""

    @abstractmethod
    def validate(self, graph: TNFRGraph) -> list[InvariantViolation]:
        """Validates invariant in the graph, returns found violations."""


class Invariant1_EPIOnlyThroughOperators(TNFRInvariant):
    """Invariant 1: EPI changes only through structural operators."""

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
                    current_epi = (
                        float(epi_value) if isinstance(epi_value, (int, float, complex)) else 0.0
                    )
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
            if not isinstance(current_epi, (int, float)) or not math.isfinite(current_epi):
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


class Invariant3_DNFRSemantics(TNFRInvariant):
    """Invariante 3: ΔNFR semantics - sign and magnitude modulate reorganization rate."""

    invariant_id = 3
    description = "ΔNFR semantics: sign and magnitude modulate reorganization rate"

    def validate(self, graph: TNFRGraph) -> list[InvariantViolation]:
        violations = []

        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            dnfr = node_data.get(DNFR_PRIMARY, 0.0)

            # Verificar que ΔNFR es un número finito
            if not isinstance(dnfr, (int, float)) or not math.isfinite(dnfr):
                violations.append(
                    InvariantViolation(
                        invariant_id=3,
                        severity=InvariantSeverity.CRITICAL,
                        description="ΔNFR is not a finite number",
                        node_id=str(node_id),
                        expected_value="finite float",
                        actual_value=f"{type(dnfr).__name__}: {dnfr}",
                        suggestion="Check operator implementation for ΔNFR calculation",
                    )
                )

            # Verificar que ΔNFR no se trata como error/loss gradient
            # (esto es más conceptual, pero podemos verificar rangos razonables)
            if isinstance(dnfr, (int, float)) and math.isfinite(dnfr):
                # ΔNFR excesivamente grande podría indicar tratamiento erróneo
                if abs(dnfr) > 1000.0:
                    violations.append(
                        InvariantViolation(
                            invariant_id=3,
                            severity=InvariantSeverity.WARNING,
                            description="ΔNFR magnitude is unusually large",
                            node_id=str(node_id),
                            expected_value="|ΔNFR| < 1000",
                            actual_value=abs(dnfr),
                            suggestion="Verify ΔNFR is not being misused as error gradient",
                        )
                    )

        return violations


class Invariant4_OperatorClosure(TNFRInvariant):
    """Invariante 4: Operator closure - composition yields valid TNFR states."""

    invariant_id = 4
    description = "Operator closure: composition yields valid TNFR states"

    def validate(self, graph: TNFRGraph) -> list[InvariantViolation]:
        violations = []

        # Verificar que el grafo mantiene estado válido después de operadores
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]

            # Verificar que los atributos esenciales existen
            required_attrs = [EPI_PRIMARY, VF_PRIMARY, THETA_PRIMARY]
            missing_attrs = [attr for attr in required_attrs if attr not in node_data]

            if missing_attrs:
                violations.append(
                    InvariantViolation(
                        invariant_id=4,
                        severity=InvariantSeverity.CRITICAL,
                        description=f"Node missing required TNFR attributes: {missing_attrs}",
                        node_id=str(node_id),
                        expected_value="All TNFR attributes present",
                        actual_value=f"Missing: {missing_attrs}",
                        suggestion="Operator composition broke TNFR state structure",
                    )
                )

        # Verificar que el grafo tiene hook ΔNFR
        if hasattr(graph, "graph"):
            if "compute_delta_nfr" not in graph.graph:
                violations.append(
                    InvariantViolation(
                        invariant_id=4,
                        severity=InvariantSeverity.WARNING,
                        description="Graph missing ΔNFR computation hook",
                        expected_value="compute_delta_nfr hook present",
                        actual_value="Hook missing",
                        suggestion="Ensure ΔNFR hook is installed for proper operator closure",
                    )
                )

        return violations


class Invariant6_NodeBirthCollapse(TNFRInvariant):
    """Invariante 6: Node birth/collapse - minimal conditions maintained."""

    invariant_id = 6
    description = "Node birth/collapse: minimal conditions maintained"

    def validate(self, graph: TNFRGraph) -> list[InvariantViolation]:
        violations = []

        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            vf = node_data.get(VF_PRIMARY, 0.0)
            dnfr = node_data.get(DNFR_PRIMARY, 0.0)

            # Extract scalar values if needed
            if isinstance(vf, dict) and "continuous" in vf:
                continue  # Skip complex structures

            if isinstance(dnfr, dict):
                continue  # Skip complex structures

            # Condiciones mínimas de nacimiento: νf suficiente
            if isinstance(vf, (int, float)) and vf < 0.001:
                violations.append(
                    InvariantViolation(
                        invariant_id=6,
                        severity=InvariantSeverity.WARNING,
                        description="Node has insufficient νf for sustained existence",
                        node_id=str(node_id),
                        expected_value="νf >= 0.001",
                        actual_value=vf,
                        suggestion="Node may be approaching collapse condition",
                    )
                )

            # Condiciones de colapso: ΔNFR extremo o νf cercano a cero
            if isinstance(dnfr, (int, float)) and math.isfinite(dnfr):
                if abs(dnfr) > 10.0:  # Dissonance extrema
                    violations.append(
                        InvariantViolation(
                            invariant_id=6,
                            severity=InvariantSeverity.WARNING,
                            description="Node experiencing extreme dissonance (collapse risk)",
                            node_id=str(node_id),
                            expected_value="|ΔNFR| < 10",
                            actual_value=abs(dnfr),
                            suggestion="High dissonance may trigger node collapse",
                        )
                    )

        return violations


class Invariant7_OperationalFractality(TNFRInvariant):
    """Invariante 7: Operational fractality - EPIs can nest without losing identity."""

    invariant_id = 7
    description = "Operational fractality: EPIs can nest without losing identity"

    def validate(self, graph: TNFRGraph) -> list[InvariantViolation]:
        violations = []

        # Verificar que estructuras EPI complejas mantienen identidad
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            epi = node_data.get(EPI_PRIMARY, 0.0)

            # Si EPI es una estructura anidada, verificar integridad
            if isinstance(epi, dict):
                # Verificar que tiene las claves esperadas para fractality
                expected_keys = {"continuous", "discrete", "grid"}
                actual_keys = set(epi.keys())

                if not actual_keys.issubset(expected_keys):
                    violations.append(
                        InvariantViolation(
                            invariant_id=7,
                            severity=InvariantSeverity.WARNING,
                            description="EPI structure has unexpected keys (fractality may be broken)",
                            node_id=str(node_id),
                            expected_value=f"Keys subset of {expected_keys}",
                            actual_value=f"Keys: {actual_keys}",
                            suggestion="Verify nested EPI structure maintains identity",
                        )
                    )

                # Verificar que los sub-EPIs tienen valores válidos
                for key in ["continuous", "discrete"]:
                    if key in epi:
                        sub_epi = epi[key]
                        if isinstance(sub_epi, (tuple, list)):
                            for val in sub_epi:
                                if isinstance(val, complex) and not math.isfinite(abs(val)):
                                    violations.append(
                                        InvariantViolation(
                                            invariant_id=7,
                                            severity=InvariantSeverity.ERROR,
                                            description=f"Sub-EPI '{key}' contains non-finite values",
                                            node_id=str(node_id),
                                            expected_value="finite values",
                                            actual_value=f"{val}",
                                            suggestion="Nested EPI identity compromised by invalid values",
                                        )
                                    )

        return violations


class Invariant8_ControlledDeterminism(TNFRInvariant):
    """Invariante 8: Controlled determinism - reproducible and traceable."""

    invariant_id = 8
    description = "Controlled determinism: reproducible and traceable"

    def validate(self, graph: TNFRGraph) -> list[InvariantViolation]:
        violations = []

        # Verificar que hay trazabilidad (history)
        if hasattr(graph, "graph"):
            config = graph.graph

            # Verificar que hay sistema de history/trace
            if "history" not in config and "HISTORY_MAXLEN" not in config:
                violations.append(
                    InvariantViolation(
                        invariant_id=8,
                        severity=InvariantSeverity.WARNING,
                        description="No history tracking configured (traceability compromised)",
                        expected_value="history or HISTORY_MAXLEN in config",
                        actual_value="Not found",
                        suggestion="Configure history tracking for reproducibility",
                    )
                )

            # Verificar que hay seed configurado para reproducibilidad
            if "RANDOM_SEED" not in config and "seed" not in config:
                violations.append(
                    InvariantViolation(
                        invariant_id=8,
                        severity=InvariantSeverity.WARNING,
                        description="No random seed configured (reproducibility at risk)",
                        expected_value="RANDOM_SEED or seed in config",
                        actual_value="Not found",
                        suggestion="Set random seed for deterministic simulations",
                    )
                )

        return violations


class Invariant9_StructuralMetrics(TNFRInvariant):
    """Invariante 9: Structural metrics - expose C(t), Si, phase, νf."""

    invariant_id = 9
    description = "Structural metrics: expose C(t), Si, phase, νf"

    def validate(self, graph: TNFRGraph) -> list[InvariantViolation]:
        violations = []

        # Verificar que los nodos exponen métricas estructurales
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]

            # Verificar métricas básicas (νf, phase ya verificados en otros invariantes)
            # Aquí verificamos métricas derivadas si existen

            # Si hay métrica Si (sense index), verificar que es válida
            if "Si" in node_data or "si" in node_data:
                si = node_data.get("Si", node_data.get("si", 0.0))
                if isinstance(si, (int, float)):
                    if not (0.0 <= si <= 1.0):
                        violations.append(
                            InvariantViolation(
                                invariant_id=9,
                                severity=InvariantSeverity.WARNING,
                                description="Sense index (Si) outside expected range",
                                node_id=str(node_id),
                                expected_value="0.0 <= Si <= 1.0",
                                actual_value=si,
                                suggestion="Verify Si calculation maintains TNFR semantics",
                            )
                        )

        # Verificar que hay métricas globales de coherencia
        if hasattr(graph, "graph"):
            config = graph.graph
            has_coherence_metric = (
                "coherence" in config or "C_t" in config or "total_coherence" in config
            )

            if not has_coherence_metric:
                violations.append(
                    InvariantViolation(
                        invariant_id=9,
                        severity=InvariantSeverity.WARNING,
                        description="No global coherence metric C(t) exposed",
                        expected_value="C(t) or coherence metric in graph",
                        actual_value="Not found",
                        suggestion="Expose total coherence C(t) for structural metrics",
                    )
                )

        return violations


class Invariant10_DomainNeutrality(TNFRInvariant):
    """Invariante 10: Domain neutrality - trans-scale and trans-domain."""

    invariant_id = 10
    description = "Domain neutrality: trans-scale and trans-domain"

    def validate(self, graph: TNFRGraph) -> list[InvariantViolation]:
        violations = []

        # Verificar que no hay hard-coded domain assumptions
        if hasattr(graph, "graph"):
            config = graph.graph

            # Buscar claves que sugieran assumptions específicas de dominio
            domain_specific_keys = [
                "physical_units",
                "meters",
                "seconds",
                "temperature",
                "biology",
                "neurons",
                "particles",
            ]

            found_domain_keys = [key for key in domain_specific_keys if key in config]

            if found_domain_keys:
                violations.append(
                    InvariantViolation(
                        invariant_id=10,
                        severity=InvariantSeverity.WARNING,
                        description=f"Domain-specific keys found: {found_domain_keys}",
                        expected_value="Domain-neutral configuration",
                        actual_value=f"Found: {found_domain_keys}",
                        suggestion="Remove domain-specific assumptions from core engine",
                    )
                )

        # Verificar que las unidades son estructurales (Hz_str, no Hz físicos)
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]

            # Si hay unidades explícitas, deben ser estructurales
            if "units" in node_data:
                units = node_data["units"]
                if isinstance(units, dict) and "vf" in units:
                    if units["vf"] not in ["Hz_str", "structural_hertz", None]:
                        violations.append(
                            InvariantViolation(
                                invariant_id=10,
                                severity=InvariantSeverity.ERROR,
                                description=f"Non-structural units for νf: {units['vf']}",
                                node_id=str(node_id),
                                expected_value="Hz_str or structural_hertz",
                                actual_value=units["vf"],
                                suggestion="Use structural units (Hz_str) not physical units",
                            )
                        )

        return violations
