"""Shared snapshot-based phase proposal for internal dynamics engines.

The proposal advances each oscillator by its live structural frequency and
adds only neighbor interactions admitted by the canonical U3 phase gate. It
never mutates the graph.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

from ..errors import TNFRValueError
from ..mathematics.unified_numerical import np

__all__ = ["propose_u3_gated_phase_step"]


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TNFRValueError(f"{name} must be a finite real scalar.")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise TNFRValueError(f"{name} must be a finite real scalar.") from exc
    if not math.isfinite(result):
        raise TNFRValueError(f"{name} must be a finite real scalar.")
    return result


def propose_u3_gated_phase_step(
    graph: Any,
    nodes: tuple[Any, ...],
    phases: Any,
    frequencies: Any,
    *,
    dt: float,
    coupling_strength: float,
) -> np.ndarray:
    """Return one simultaneous free-advance plus U3-gated phase proposal."""
    from ..operators._phase_gate import (
        U3PhaseGateError,
        resolve_u3_phase_neighbors,
    )

    if tuple(graph.nodes()) != tuple(nodes):
        raise TNFRValueError("Phase proposal node order differs from the graph.")
    phase = np.asarray(phases, dtype=float)
    frequency = np.asarray(frequencies, dtype=float)
    count = len(nodes)
    if phase.shape != (count,) or frequency.shape != (count,):
        raise TNFRValueError("Phase and frequency vectors must match node order.")
    if not np.all(np.isfinite(phase)) or not np.all(np.isfinite(frequency)):
        raise TNFRValueError("Phase and frequency vectors must be finite.")
    if np.any(frequency < 0.0):
        raise TNFRValueError("Structural frequency must be nonnegative.")

    time_step = _finite_real(dt, "phase integration dt")
    strength = _finite_real(coupling_strength, "phase coupling strength")
    if time_step <= 0.0:
        raise TNFRValueError("phase integration dt must be positive.")
    if strength < 0.0:
        raise TNFRValueError("phase coupling strength must be nonnegative.")

    index = {node: offset for offset, node in enumerate(nodes)}
    with np.errstate(over="ignore", invalid="ignore"):
        proposal = phase + time_step * frequency
    if not np.all(np.isfinite(proposal)):
        raise TNFRValueError("Free phase proposal must remain finite.")
    for offset, node in enumerate(nodes):
        try:
            gate = resolve_u3_phase_neighbors(
                graph.graph,
                phase[offset],
                graph.neighbors(node),
                phase_getter=lambda neighbor: phase[index[neighbor]],
                operator_code="UM",
                require_compatible=False,
            )
        except U3PhaseGateError as exc:
            raise TNFRValueError(
                "Phase proposal failed the U3 phase gate.",
                context={"node": node, "reason": exc.failed_condition},
            ) from exc

        if gate.neighbors:
            coupling = math.fsum(
                math.sin(neighbor_phase - gate.target_phase)
                for neighbor_phase in gate.phases
            )
            with np.errstate(over="ignore", invalid="ignore"):
                proposal[offset] += (
                    time_step * strength * coupling / len(gate.neighbors)
                )
            if not math.isfinite(float(proposal[offset])):
                raise TNFRValueError("Coupled phase proposal must remain finite.")

    wrapped = np.mod(proposal, 2.0 * math.pi)
    if not np.all(np.isfinite(wrapped)):
        raise TNFRValueError("Wrapped phase proposal must remain finite.")
    return wrapped
