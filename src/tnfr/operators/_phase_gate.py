"""Shared non-mutating U3 phase gate for Coupling and Resonance."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from ..constants.canonical import DELTA_PHI_MAX
from ..mathematics.unified_numerical import np
from ..utils import angle_diff

__all__ = [
    "U3PhaseGateError",
    "U3PhaseNeighborSet",
    "phase_limit_is_canonical",
    "resolve_u3_phase_limits",
    "resolve_u3_phase_neighbors",
    "select_u3_phase_neighbors",
]


class U3PhaseGateError(ValueError):
    """Invalid U3 configuration, phase state, or compatible-neighbor set."""

    def __init__(self, message: str, *, failed_condition: str) -> None:
        super().__init__(message)
        self.failed_condition = failed_condition


@dataclass(frozen=True)
class U3PhaseNeighborSet:
    """Validated target phase and its U3-compatible neighbor subset."""

    canonical_graph_limit: float
    effective_limit: float
    target_phase: float
    neighbors: tuple[Any, ...]
    phases: tuple[float, ...]


def _finite_real(value: Any, label: str, condition: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise U3PhaseGateError(
            f"{label} must be a finite real scalar, not boolean",
            failed_condition=condition,
        )
    try:
        resolved = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise U3PhaseGateError(
            f"{label} must be a finite real scalar",
            failed_condition=condition,
        ) from exc
    if not math.isfinite(resolved):
        raise U3PhaseGateError(
            f"{label} must be a finite real scalar",
            failed_condition=condition,
        )
    return resolved


def phase_limit_is_canonical(
    limit: Any, canonical_limit: float = DELTA_PHI_MAX
) -> bool:
    """Return whether a proposed hard phase limit lies in ``[0, pi/2]``."""
    try:
        resolved = _finite_real(limit, "DELTA_PHI_MAX", "phase_limit")
    except U3PhaseGateError:
        return False
    return 0.0 <= resolved <= float(canonical_limit)


def resolve_u3_phase_limits(
    graph_attributes: Mapping[str, Any],
    *,
    operator_code: str,
) -> tuple[float, float]:
    """Resolve the hard graph gate and an optional UM-only tightening.

    ``DELTA_PHI_MAX`` is the non-negotiable U3 limit and must lie in the
    canonical interval ``[0, pi/2]``. ``UM_MAX_PHASE_DIFF`` may be smaller;
    larger values are accepted as compatibility configuration but are clamped
    by the hard graph gate and therefore can never weaken it.
    """
    hard_limit = _finite_real(
        graph_attributes.get("DELTA_PHI_MAX", DELTA_PHI_MAX),
        "DELTA_PHI_MAX",
        "phase_limit",
    )
    if not 0.0 <= hard_limit <= float(DELTA_PHI_MAX):
        raise U3PhaseGateError(
            "DELTA_PHI_MAX must lie in the canonical interval "
            f"[0, {DELTA_PHI_MAX}]",
            failed_condition="phase_limit",
        )

    effective_limit = hard_limit
    if operator_code == "UM" and "UM_MAX_PHASE_DIFF" in graph_attributes:
        um_limit = _finite_real(
            graph_attributes["UM_MAX_PHASE_DIFF"],
            "UM_MAX_PHASE_DIFF",
            "phase_limit",
        )
        if um_limit < 0.0:
            raise U3PhaseGateError(
                "UM_MAX_PHASE_DIFF must be nonnegative",
                failed_condition="phase_limit",
            )
        effective_limit = min(hard_limit, um_limit)
    return hard_limit, effective_limit


def select_u3_phase_neighbors(
    target_phase: Any,
    neighbors: Iterable[Any],
    *,
    phase_getter: Callable[[Any], Any],
    phase_limit: float,
    require_compatible: bool,
) -> tuple[float, tuple[Any, ...], tuple[float, ...]]:
    """Validate phases and return only neighbors inside ``phase_limit``."""
    resolved_target = _finite_real(
        target_phase, "target phase", "target_phase"
    )
    candidates = tuple(neighbors)
    compatible: list[Any] = []
    phases: list[float] = []
    for neighbor in candidates:
        try:
            raw_phase = phase_getter(neighbor)
        except (AttributeError, KeyError, OverflowError, TypeError, ValueError) as exc:
            raise U3PhaseGateError(
                "neighbor phases must be finite real scalars",
                failed_condition="neighbor_phase",
            ) from exc
        neighbor_phase = _finite_real(
            raw_phase, "neighbor phase", "neighbor_phase"
        )
        if abs(angle_diff(resolved_target, neighbor_phase)) <= phase_limit:
            compatible.append(neighbor)
            phases.append(neighbor_phase)

    if require_compatible and not compatible:
        if candidates:
            message = (
                "U3 phase gate rejected the operation: no compatible neighbor "
                f"within {phase_limit} radians"
            )
        else:
            message = (
                "U3 phase gate requires at least one coupled neighbor "
                "inside the effective phase limit"
            )
        raise U3PhaseGateError(
            message, failed_condition="u3_phase_compatibility"
        )
    return resolved_target, tuple(compatible), tuple(phases)


def resolve_u3_phase_neighbors(
    graph_attributes: Mapping[str, Any],
    target_phase: Any,
    neighbors: Iterable[Any],
    *,
    phase_getter: Callable[[Any], Any],
    operator_code: str,
    require_compatible: bool = True,
) -> U3PhaseNeighborSet:
    """Resolve limits and return a validated compatible-neighbor snapshot."""
    hard_limit, effective_limit = resolve_u3_phase_limits(
        graph_attributes, operator_code=operator_code
    )
    resolved_target, compatible, phases = select_u3_phase_neighbors(
        target_phase,
        neighbors,
        phase_getter=phase_getter,
        phase_limit=effective_limit,
        require_compatible=require_compatible,
    )
    return U3PhaseNeighborSet(
        canonical_graph_limit=hard_limit,
        effective_limit=effective_limit,
        target_phase=resolved_target,
        neighbors=compatible,
        phases=phases,
    )
