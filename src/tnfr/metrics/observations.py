"""Optional provenance envelopes for TNFR structural readouts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from .._exact_time import finite_represented_real, nonnegative_represented_time

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
    """Detach a graph tetrad, retaining summary versus per-node aggregation.

    Availability describes the supplied payload, not authenticated graph evidence.
    A partial snapshot is useful but incomplete; no available canonical channel
    marks the whole envelope unavailable. Extended currents are not required for
    canonical tetrad completeness.
    """
    summary = (
        isinstance(snapshot, Mapping)
        and {"phi_s", "phase_grad", "phase_curv", "xi_c_available"} <= snapshot.keys()
    )
    field_names = ("phi_s", "grad_phi", "k_phi")
    if summary:
        availability = {
            name: isinstance(snapshot[key], Mapping)
            and snapshot[key].get("available") is True
            and all(
                _available_scalar(snapshot[key].get(statistic))
                for statistic in ("mean", "min", "max", "std")
            )
            for name, key in zip(field_names, ("phi_s", "phase_grad", "phase_curv"))
        }
        availability["xi_c"] = snapshot["xi_c_available"] is True and _available_scalar(
            snapshot.get("xi_c"), positive=True
        )
        complete = all(availability.values())
        aggregation = "per_node_field_statistics_plus_global_xi_c"
    else:
        local_fields = {
            name: (
                snapshot.get(name)
                if isinstance(snapshot, Mapping)
                else getattr(snapshot, name, None)
            )
            for name in field_names
        }
        availability = {
            name: _available_field(values) for name, values in local_fields.items()
        }
        xi = (
            snapshot.get("xi_c")
            if isinstance(snapshot, Mapping)
            else getattr(snapshot, "xi_c", None)
        )
        availability["xi_c"] = _available_scalar(xi, positive=True)
        complete = all(availability.values()) and all(
            set(values) == set(local_fields["phi_s"])
            for values in local_fields.values()
        )
        aggregation = "per_node_fields_plus_global_xi_c"
    labels = []
    for name in (() if summary else ("phi_s", "grad_phi", "k_phi", "j_phi", "j_dnfr")):
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
        aggregation=aggregation,
        derivative_kind="read_only_snapshot",
        equilibrium_tolerance=tolerance,
        scope="canonical graph tetrad telemetry",
        value=snapshot,
        metadata={
            "unavailable": not any(availability.values()),
            "complete": complete,
            "field_availability": availability,
        },
        _identity_labels=tuple(labels),
    )


def _available_scalar(value: Any, *, positive: bool = False) -> bool:
    try:
        numeric = finite_represented_real(value, "tetrad value")[0]
    except (TypeError, ValueError):
        return False
    return not positive or numeric > 0.0


def _available_field(values: Any) -> bool:
    return (
        isinstance(values, Mapping)
        and bool(values)
        and all(_available_scalar(value) for value in values.values())
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
