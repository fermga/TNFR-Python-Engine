"""Shared snapshot-based, configured phase model for internal engines.

This model identifies supplied structural capacity with a free angular rate:
``theta_next = theta + dt*frequency + dt*coupling`` modulo ``2*pi``.
The neighbor sine term uses only U3-admitted neighbors. This identification
and the coupling strength are constitutive premises, not deductions from
``dEPI/dt = nu_f*DeltaNFR``. The arithmetic uses radians per supplied time unit;
it does not infer cycles per second or apply the optional physical Hz bridge.

The nodal optimizer and FFT engine share this model. Ordinary runtime phase
coordination instead uses its separately configured relaxation map. Neither
law may be substituted for the other based only on the nodal EPI identity.
Both optimized consumers pair this proposal with isolated EPI diffusion, whose
pressure does not read phase. The explicit, opt-in composition in
``physics.p2_phase_form`` instead combines this proposal with fresh multichannel
pressure on fixed P2. It retains the same supplied angular-rate premise.
See ``theory/FORCED_SUPPORT_BALANCE.md`` sections 23 and 26. This helper returns
a proposal without mutating the graph.
"""

from __future__ import annotations

import math
from typing import Any

from .._exact_time import finite_represented_real
from ..errors import TNFRValueError
from ..mathematics.unified_numerical import np

__all__ = ["propose_u3_gated_phase_step"]


def _finite_real(value: Any, name: str) -> float:
    try:
        return finite_represented_real(value, name)[0]
    except (TypeError, ValueError) as exc:
        raise TNFRValueError(str(exc)) from exc


def _finite_vector(values: Any, count: int, name: str) -> np.ndarray:
    """Validate original scalar kinds before binary64 array conversion."""
    try:
        original = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise TNFRValueError(f"{name} vector must match node order.") from exc
    if original.shape != (count,):
        raise TNFRValueError("Phase and frequency vectors must match node order.")
    return np.array(
        [
            _finite_real(value, f"{name}[{index}]")
            for index, value in enumerate(original)
        ],
        dtype=float,
    )


def propose_u3_gated_phase_step(
    graph: Any,
    nodes: tuple[Any, ...],
    phases: Any,
    frequencies: Any,
    *,
    dt: float,
    coupling_strength: float,
) -> np.ndarray:
    """Return one simultaneous free-advance plus U3-gated phase proposal.

    Phase and capacity vectors contain finite real scalars, not Boolean,
    text or complex values. Validation precedes binary64 conversion; no
    imaginary component is discarded from an inverse spectral transform.
    Nonzero input scalars must remain nonzero when materialized as binary64.
    """
    if tuple(graph.nodes()) != tuple(nodes):
        raise TNFRValueError("Phase proposal node order differs from the graph.")
    return _propose_u3_phase_from_neighbors(
        graph.graph,
        nodes,
        graph.neighbors,
        phases,
        frequencies,
        dt=dt,
        coupling_strength=coupling_strength,
    )


def _propose_u3_phase_from_neighbors(
    graph_attributes,
    nodes,
    neighbors,
    phases,
    frequencies,
    *,
    dt,
    coupling_strength,
):
    """Shared arithmetic for live unique support and observed counted support.

    Internal callers supply the node order and neighbor reader. In a counted
    quotient, repeated indices stand for distinct fine neighbors; an internal
    neighbor contributes to the admitted denominator even when its sine is zero.
    Graph metadata and observed multiplicities retain their separate owners.
    """
    from ..operators._phase_gate import U3PhaseGateError, resolve_u3_phase_neighbors

    count = len(nodes)
    phase = _finite_vector(phases, count, "phase")
    frequency = _finite_vector(frequencies, count, "structural frequency")
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
                graph_attributes,
                phase[offset],
                neighbors(node),
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
