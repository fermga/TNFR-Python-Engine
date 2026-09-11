"""Optional provenance envelopes for TNFR structural readouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

__all__ = [
    "StructuralObservation",
    "observe_graph_tetrad",
    "observe_arithmetic_nfr",
    "observe_emergent_element",
]


@dataclass(frozen=True)
class StructuralObservation:
    """Read-only metadata describing how a structural value was obtained."""

    domain: str
    pressure_realization: str
    aggregation: str
    derivative_kind: str
    equilibrium_tolerance: float | None
    scope: str
    value: Any = None
    metadata: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        if not self.domain or not self.pressure_realization:
            raise ValueError("domain and pressure_realization are required")
        if not self.aggregation or not self.derivative_kind or not self.scope:
            raise ValueError("observation provenance fields are required")
        if (
            self.equilibrium_tolerance is not None
            and self.equilibrium_tolerance < 0
        ):
            raise ValueError("equilibrium_tolerance must be nonnegative")
        object.__setattr__(
            self, "metadata", MappingProxyType(dict(self.metadata))
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a detached serializable representation."""
        return {
            "domain": self.domain,
            "pressure_realization": self.pressure_realization,
            "aggregation": self.aggregation,
            "derivative_kind": self.derivative_kind,
            "equilibrium_tolerance": self.equilibrium_tolerance,
            "scope": self.scope,
            "value": self.value,
            "metadata": dict(self.metadata),
        }


def observe_graph_tetrad(
    snapshot: Any, *, tolerance: float | None = None
) -> StructuralObservation:
    """Wrap a graph tetrad snapshot without replacing its result type."""
    return StructuralObservation(
        domain="graph",
        pressure_realization="graph_coupled_delta_nfr",
        aggregation="per_node_fields_plus_global_xi_c",
        derivative_kind="read_only_snapshot",
        equilibrium_tolerance=tolerance,
        scope="canonical graph tetrad telemetry",
        value=snapshot,
        metadata={"unavailable": snapshot is None},
    )


def observe_arithmetic_nfr(
    readout: Mapping[str, Any]
) -> StructuralObservation:
    """Wrap an arithmetic NFR readout with explicit aggregate semantics."""
    unavailable = (
        not bool(readout)
        or readout.get("readout_available") is False
        or readout.get("coherence") is None
    )
    return StructuralObservation(
        domain="arithmetic",
        pressure_realization="arithmetic_divisor_pressure",
        aggregation="canonical_static_coherence_of_mean_absolute_pressure",
        derivative_kind="static_zero_readout",
        equilibrium_tolerance=1e-12,
        scope="descriptive arithmetic fixed-point observation",
        value=dict(readout),
        metadata={
            "unavailable": unavailable,
            "canonical_field": "coherence",
            "descriptive_field": "mean_local_coherence",
        },
    )


def observe_emergent_element(element: Any) -> StructuralObservation:
    """Wrap a chemical closed-shell readout without claiming graph dynamics."""
    value = element.as_dict() if hasattr(element, "as_dict") else element
    return StructuralObservation(
        domain="chemical",
        pressure_realization="valence_shell_distance",
        aggregation="single_element_readout",
        derivative_kind="static_zero_readout",
        equilibrium_tolerance=1e-12,
        scope="configured shell-filling chemical observation",
        value=value,
        metadata={"unavailable": element is None},
    )
