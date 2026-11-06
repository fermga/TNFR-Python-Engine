"""TNFR Invariant Validators.

This module retains ten historical validation check identifiers for API and
report compatibility. AGENTS.md defines six canonical invariants; the checks
below split their implementation concerns and do not define a second canon.

Legacy check identifiers:
1. Signed form, configured bounds and optional operator-history marker
2. Nonnegative structural capacity and configured bounds
3. Signed pressure admission and magnitude advisory
4. Required attributes and pressure-hook metadata
5. Circular phase admission and an advisory edge threshold
6. Capacity and pressure activity advisories
7. Operational fractality: EPIs can nest without losing identity
8. Controlled determinism: reproducible and traceable
9. Structural metrics: expose C(t), Si, phase, νf
10. Domain neutrality: trans-scale and trans-domain
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .._exact_time import bounded_represented_real, finite_represented_real
from ..alias import get_attr
from ..constants import DEFAULTS, EPI_PRIMARY
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_SI, ALIAS_THETA, ALIAS_VF
from ..constants.canonical import DELTA_PHI_MAX
from ..types import NodeId, TNFRGraph, require_finite_real_scalar_epi
from ..utils import angle_diff

# ---------------------------------------------------------------------------
# Invariant guardrail thresholds
# ---------------------------------------------------------------------------
_VF_MIN_SUSTAINED = 0.001
_DNFR_MAGNITUDE_WARNING = 1000.0
_DNFR_EXTREME_DISSONANCE = 10.0

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
    node_id: NodeId | None = None
    expected_value: Any | None = None
    actual_value: Any | None = None
    suggestion: str | None = None


def _node_scalar(
    graph: TNFRGraph,
    node: NodeId,
    aliases: tuple[str, ...],
    label: str,
    violations: list[InvariantViolation],
    invariant_id: int,
    *,
    signed_form: bool = False,
    nonnegative: bool = False,
    severity: InvariantSeverity = InvariantSeverity.CRITICAL,
) -> float | None:
    """Admit one authoritative scalar or record a failed observation.

    Absent channels retain the legacy zero default; check 4 separately reports
    missing required attributes. An invalid present alias never falls through.
    """
    try:
        raw = get_attr(
            graph.nodes[node], aliases, 0.0, strict=True, conv=lambda value: value
        )
        if signed_form:
            return require_finite_real_scalar_epi(raw, label)
        return bounded_represented_real(
            raw, label, minimum=0.0 if nonnegative else None
        )[0]
    except (TypeError, ValueError, OverflowError) as exc:
        violations.append(
            InvariantViolation(
                invariant_id=invariant_id,
                severity=severity,
                description=f"{label} has an invalid scalar domain",
                node_id=str(node),
                expected_value=(
                    "finite uniform-real signed form"
                    if signed_form
                    else (
                        "finite nonnegative real scalar"
                        if nonnegative
                        else "finite real scalar"
                    )
                ),
                actual_value=str(exc),
                suggestion="Correct the authoritative source value without projection",
            )
        )
        return None


def _configured_bounds(graph: TNFRGraph, low: str, high: str) -> tuple[float, float]:
    """Read finite ordered diagnostic bounds without creating physical laws."""
    config = graph.graph
    lower = finite_represented_real(config.get(low, DEFAULTS[low]), low)[0]
    upper = finite_represented_real(config.get(high, DEFAULTS[high]), high)[0]
    if lower > upper:
        raise ValueError(f"{low} must not exceed {high}")
    return lower, upper


class TNFRInvariant(ABC):
    """Base class for TNFR invariant validators."""

    @property
    @abstractmethod
    def invariant_id(self) -> int:
        """Legacy validation check number (1-10)."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the invariant."""

    @abstractmethod
    def validate(self, graph: TNFRGraph) -> list[InvariantViolation]:
        """Validates invariant in the graph, returns found violations."""


class Invariant1_EPIOnlyThroughOperators(TNFRInvariant):
    """Legacy check 1: signed form, configured bounds and optional history marker.

    The retained class name does not prohibit declared continuous nodal flow.
    The optional operator marker alone cannot determine a change's provenance.
    """

    invariant_id = 1
    description = "Signed form and optional operator-history diagnostic"

    def __init__(self) -> None:
        self._previous_epi_values: dict[Any, float] = {}

    def validate(self, graph: TNFRGraph) -> list[InvariantViolation]:
        violations: list[InvariantViolation] = []
        epi_min, epi_max = _configured_bounds(graph, "EPI_MIN", "EPI_MAX")
        admitted = {}
        for node_id in graph:
            current = _node_scalar(
                graph, node_id, ALIAS_EPI, "EPI", violations, 1, signed_form=True
            )
            if current is None:
                continue
            admitted[node_id] = current
            if not epi_min <= current <= epi_max:
                violations.append(
                    InvariantViolation(
                        invariant_id=1,
                        severity=InvariantSeverity.ERROR,
                        description=f"EPI outside configured bounds [{epi_min}, {epi_max}]",
                        node_id=str(node_id),
                        expected_value=f"{epi_min} <= EPI <= {epi_max}",
                        actual_value=current,
                        suggestion="Review the configured bounds for the declared model",
                    )
                )
            previous = self._previous_epi_values.get(node_id)
            if (
                previous is not None
                and getattr(graph, "_last_operator_applied", None) is False
                and not math.isclose(current, previous, rel_tol=0.0, abs_tol=1e-10)
            ):
                violations.append(
                    InvariantViolation(
                        invariant_id=1,
                        severity=InvariantSeverity.WARNING,
                        description="EPI changed while the optional operator marker was false",
                        node_id=str(node_id),
                        expected_value=previous,
                        actual_value=current,
                        suggestion="Check the declared flow or event record; this marker does not identify the cause",
                    )
                )
        self._previous_epi_values = admitted
        return violations


class Invariant2_VfInHzStr(TNFRInvariant):
    """Legacy check 2: nonnegative capacity and configured range policy."""

    invariant_id = 2
    description = "Nonnegative structural capacity and configured bounds"

    def validate(self, graph: TNFRGraph) -> list[InvariantViolation]:
        violations: list[InvariantViolation] = []
        vf_min, vf_max = _configured_bounds(graph, "VF_MIN", "VF_MAX")
        for node_id in graph:
            capacity = _node_scalar(
                graph, node_id, ALIAS_VF, "Capacity", violations, 2, nonnegative=True
            )
            if capacity is not None and not vf_min <= capacity <= vf_max:
                violations.append(
                    InvariantViolation(
                        invariant_id=2,
                        severity=InvariantSeverity.ERROR,
                        description=f"Capacity outside configured bounds [{vf_min}, {vf_max}]",
                        node_id=str(node_id),
                        expected_value=f"{vf_min} <= capacity <= {vf_max}",
                        actual_value=capacity,
                        suggestion="Review the configured capacity bounds; zero is admitted by the unforced nodal row",
                    )
                )
        return violations


class Invariant5_ExplicitPhaseChecks(TNFRInvariant):
    """Legacy check 5: admitted circular phases and an advisory edge threshold."""

    invariant_id = 5
    description = "Circular phase observations under a selected advisory threshold"

    def __init__(self, phase_coupling_threshold: float = DELTA_PHI_MAX) -> None:
        self.phase_coupling_threshold = bounded_represented_real(
            phase_coupling_threshold, "phase coupling threshold", minimum=0.0
        )[0]

    def validate(self, graph: TNFRGraph) -> list[InvariantViolation]:
        threshold = bounded_represented_real(
            self.phase_coupling_threshold, "phase coupling threshold", minimum=0.0
        )[0]
        violations: list[InvariantViolation] = []
        phases = {}
        for node_id in graph:
            phase = _node_scalar(graph, node_id, ALIAS_THETA, "Phase", violations, 5)
            if phase is None:
                continue
            phases[node_id] = phase
            if not 0.0 <= phase <= math.tau:
                violations.append(
                    InvariantViolation(
                        invariant_id=5,
                        severity=InvariantSeverity.WARNING,
                        description="Phase representative outside the [0, 2pi] display convention",
                        node_id=str(node_id),
                        expected_value="Optional display convention only",
                        actual_value=phase,
                        suggestion="Retain the real phase lift when the declared model requires it",
                    )
                )
        for node1, node2 in graph.edges():
            if node1 not in phases or node2 not in phases:
                continue
            try:
                phase_diff = abs(angle_diff(phases[node1], phases[node2]))
            except (TypeError, ValueError, OverflowError) as exc:
                violations.append(
                    InvariantViolation(
                        invariant_id=5,
                        severity=InvariantSeverity.ERROR,
                        description="Phase difference is not representable",
                        node_id=f"{node1}-{node2}",
                        actual_value=str(exc),
                    )
                )
                continue
            if phase_diff > threshold:
                violations.append(
                    InvariantViolation(
                        invariant_id=5,
                        severity=InvariantSeverity.WARNING,
                        description="Large phase difference between coupled nodes",
                        node_id=f"{node1}-{node2}",
                        expected_value=f"<= {threshold}",
                        actual_value=phase_diff,
                        suggestion="Review the declared coupling policy; this observation does not execute U3 admission",
                    )
                )
        return violations


class Invariant3_DNFRSemantics(TNFRInvariant):
    """Legacy check 3: finite signed pressure and a magnitude advisory."""

    invariant_id = 3
    description = "Signed pressure admission and diagnostic magnitude policy"

    def validate(self, graph: TNFRGraph) -> list[InvariantViolation]:
        violations: list[InvariantViolation] = []
        for node_id in graph:
            pressure = _node_scalar(
                graph, node_id, ALIAS_DNFR, "DeltaNFR", violations, 3
            )
            if pressure is not None and abs(pressure) > _DNFR_MAGNITUDE_WARNING:
                violations.append(
                    InvariantViolation(
                        invariant_id=3,
                        severity=InvariantSeverity.WARNING,
                        description="DeltaNFR exceeds the diagnostic magnitude threshold",
                        node_id=str(node_id),
                        expected_value=f"|DeltaNFR| <= {_DNFR_MAGNITUDE_WARNING}",
                        actual_value=abs(pressure),
                        suggestion="Review this advisory against the declared pressure law and units",
                    )
                )
        return violations


class Invariant4_OperatorClosure(TNFRInvariant):
    """Legacy check 4: attribute and pressure-hook presence, not a closure proof."""

    invariant_id = 4
    description = "Required nodal attributes and configured pressure-hook metadata"

    def validate(self, graph: TNFRGraph) -> list[InvariantViolation]:
        violations = []

        # Verify that graph maintains valid state after operators
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]

            # Verify essential attributes exist
            required_attrs = (ALIAS_EPI, ALIAS_VF, ALIAS_THETA)
            missing_attrs = [
                aliases[0]
                for aliases in required_attrs
                if not any(alias in node_data for alias in aliases)
            ]

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

        # Verify the graph has a ΔNFR hook
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
    """Legacy check 6: activity advisories, not existence or collapse criteria."""

    invariant_id = 6
    description = "Capacity and pressure activity advisories"

    def validate(self, graph: TNFRGraph) -> list[InvariantViolation]:
        violations: list[InvariantViolation] = []
        for node_id in graph:
            capacity = _node_scalar(
                graph, node_id, ALIAS_VF, "Capacity", violations, 6, nonnegative=True
            )
            pressure = _node_scalar(
                graph, node_id, ALIAS_DNFR, "DeltaNFR", violations, 6
            )
            if capacity is not None and capacity < _VF_MIN_SUSTAINED:
                violations.append(
                    InvariantViolation(
                        invariant_id=6,
                        severity=InvariantSeverity.WARNING,
                        description="Capacity below the selected activity threshold",
                        node_id=str(node_id),
                        expected_value=f"capacity >= {_VF_MIN_SUSTAINED} for this advisory",
                        actual_value=capacity,
                        suggestion="Zero capacity freezes unforced form; this threshold does not decide NFR existence",
                    )
                )
            if pressure is not None and abs(pressure) > _DNFR_EXTREME_DISSONANCE:
                violations.append(
                    InvariantViolation(
                        invariant_id=6,
                        severity=InvariantSeverity.WARNING,
                        description="Pressure exceeds the selected activity threshold",
                        node_id=str(node_id),
                        expected_value=f"|DeltaNFR| <= {_DNFR_EXTREME_DISSONANCE} for this advisory",
                        actual_value=abs(pressure),
                        suggestion="A collapse event requires its own admission and occurrence law",
                    )
                )
        return violations


class Invariant7_OperationalFractality(TNFRInvariant):
    """Invariante 7: Operational fractality - EPIs can nest without losing identity."""

    invariant_id = 7
    description = "Operational fractality: EPIs can nest without losing identity"

    def validate(self, graph: TNFRGraph) -> list[InvariantViolation]:
        violations = []

        # Verify complex EPI structures maintain identity
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            epi = node_data.get(EPI_PRIMARY, 0.0)

            # If EPI is a nested structure, verify integrity
            if isinstance(epi, dict):
                # Verify it has the expected keys for fractality
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

                # Verify that sub-EPIs have valid values
                for key in ["continuous", "discrete"]:
                    if key in epi:
                        sub_epi = epi[key]
                        if isinstance(sub_epi, (tuple, list)):
                            for val in sub_epi:
                                if isinstance(val, complex) and not math.isfinite(
                                    abs(val)
                                ):
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

        # Verify traceability (history)
        if hasattr(graph, "graph"):
            config = graph.graph

            # Verify there is a history/trace system
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

            # Verify seed is configured for reproducibility
            if "RANDOM_SEED" not in config and "seed" not in config:
                violations.append(
                    InvariantViolation(
                        invariant_id=8,
                        severity=InvariantSeverity.WARNING,
                        description="No random seed configured (reproducibility at risk)",
                        expected_value="RANDOM_SEED or seed in config",
                        actual_value="Not found",
                        suggestion="set random seed for deterministic simulations",
                    )
                )

        return violations


class Invariant9_StructuralMetrics(TNFRInvariant):
    """Historical check 9: expose valid C(t), Si, phase and νf telemetry."""

    invariant_id = 9
    description = "Structural metrics: expose C(t), Si, phase, νf"

    def validate(self, graph: TNFRGraph) -> list[InvariantViolation]:
        violations: list[InvariantViolation] = []

        # Missing Si is unavailable; a present source must retain its domain.
        for node_id, node_data in graph.nodes(data=True):
            if any(alias in node_data for alias in ALIAS_SI):
                _node_scalar(
                    graph,
                    node_id,
                    ALIAS_SI,
                    "Sense index (Si)",
                    violations,
                    9,
                    nonnegative=True,
                    severity=InvariantSeverity.WARNING,
                )

        if not hasattr(graph, "graph"):
            return violations

        config = graph.graph
        exposed: list[tuple[str, Any]] = [
            (key, config[key])
            for key in ("coherence", "C_t", "total_coherence")
            if key in config
        ]
        history = config.get("history")
        if isinstance(history, Mapping):
            samples = history.get("C_steps")
            if (
                isinstance(samples, Sequence)
                and not isinstance(samples, (str, bytes))
                and samples
            ):
                exposed.append(("history.C_steps[-1]", samples[-1]))

        if not exposed:
            violations.append(
                InvariantViolation(
                    invariant_id=9,
                    severity=InvariantSeverity.WARNING,
                    description="No global coherence metric C(t) exposed",
                    expected_value="canonical C(t) telemetry",
                    actual_value="Not found",
                    suggestion="Expose compute_coherence(graph) in telemetry",
                )
            )
            return violations

        from .._coherence_validation import validate_structural_coherence
        from ..metrics.common import compute_coherence

        expected = float(compute_coherence(graph))
        for source, raw_value in exposed:
            try:
                measured = validate_structural_coherence(
                    raw_value, name=f"C(t) from {source}"
                )
            except (TypeError, ValueError) as exc:
                violations.append(
                    InvariantViolation(
                        invariant_id=9,
                        severity=InvariantSeverity.ERROR,
                        description="Exposed C(t) has an invalid scalar domain",
                        expected_value="finite value in [0, 1]",
                        actual_value=raw_value,
                        suggestion=str(exc),
                    )
                )
                continue
            if not math.isclose(measured, expected, rel_tol=1.0e-9, abs_tol=1.0e-12):
                violations.append(
                    InvariantViolation(
                        invariant_id=9,
                        severity=InvariantSeverity.WARNING,
                        description="Exposed C(t) does not match canonical channels",
                        expected_value=expected,
                        actual_value={"source": source, "value": measured},
                        suggestion=(
                            "Recompute C(t) from mean |DeltaNFR| and mean |dEPI/dt|"
                        ),
                    )
                )

        return violations


class Invariant10_DomainNeutrality(TNFRInvariant):
    """Invariante 10: Domain neutrality - trans-scale and trans-domain."""

    invariant_id = 10
    description = "Domain neutrality: trans-scale and trans-domain"

    def validate(self, graph: TNFRGraph) -> list[InvariantViolation]:
        violations = []

        # Verify no hard-coded domain assumptions
        if hasattr(graph, "graph"):
            config = graph.graph

            # Search for keys suggesting domain-specific assumptions
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

        # Verify that units are structural (Hz_str, not physical Hz)
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]

            # If explicit units exist, they must be structural
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
