"""Optional provenance envelopes for TNFR structural readouts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from .._exact_time import nonnegative_represented_time

__all__ = [
    "StructuralObservation",
    "observe_graph_tetrad",
    "observe_arithmetic_nfr",
]


@dataclass(frozen=True)
class StructuralObservation:
    """Detached values with fixed provenance and a read-only metadata mapping.

    Payloads must support deep copying. Nested payloads retain their Python
    types; they are not recursively frozen or automatically JSON encoded.
    """

    domain: str
    pressure_realization: str
    aggregation: str
    derivative_kind: str
    equilibrium_tolerance: float | None
    scope: str
    value: Any = None
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    _identity_labels: tuple[Any, ...] = field(default=(), repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in (
            "domain",
            "pressure_realization",
            "aggregation",
            "derivative_kind",
            "scope",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if self.equilibrium_tolerance is not None:
            tolerance, _ = nonnegative_represented_time(
                self.equilibrium_tolerance, "equilibrium_tolerance"
            )
            object.__setattr__(self, "equilibrium_tolerance", tolerance)
        if not isinstance(self.metadata, Mapping):
            raise TypeError("observation metadata must be a mapping")
        object.__setattr__(self, "_identity_labels", tuple(self._identity_labels))
        memo = {id(label): label for label in self._identity_labels}
        object.__setattr__(self, "value", deepcopy(self.value, memo))
        object.__setattr__(
            self, "metadata", MappingProxyType(deepcopy(dict(self.metadata), memo))
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a detached payload, preserving its Python value types."""
        memo = {id(label): label for label in self._identity_labels}
        return {
            "domain": self.domain,
            "pressure_realization": self.pressure_realization,
            "aggregation": self.aggregation,
            "derivative_kind": self.derivative_kind,
            "equilibrium_tolerance": self.equilibrium_tolerance,
            "scope": self.scope,
            "value": deepcopy(self.value, memo),
            "metadata": deepcopy(dict(self.metadata), memo),
        }


def observe_graph_tetrad(
    snapshot: Any, *, tolerance: float | None = None
) -> StructuralObservation:
    """Detach a graph tetrad without replacing its type or opaque node labels."""
    labels = []
    for name in ("phi_s", "grad_phi", "k_phi", "j_phi", "j_dnfr"):
        values = (
            snapshot.get(name)
            if isinstance(snapshot, Mapping)
            else getattr(snapshot, name, None)
        )
        if isinstance(values, Mapping):
            labels.extend(values)
    return StructuralObservation(
        domain="graph",
        pressure_realization="graph_coupled_delta_nfr",
        aggregation="per_node_fields_plus_global_xi_c",
        derivative_kind="read_only_snapshot",
        equilibrium_tolerance=tolerance,
        scope="canonical graph tetrad telemetry",
        value=snapshot,
        metadata={"unavailable": snapshot is None},
        _identity_labels=tuple(labels),
    )


def observe_arithmetic_nfr(readout: Mapping[str, Any]) -> StructuralObservation:
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
