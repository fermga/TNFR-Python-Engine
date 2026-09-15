"""Shared signed THOL pressure proposal, without hierarchy or flow effects."""

from dataclasses import dataclass
import math

from ..constants.aliases import ALIAS_DNFR
from ..errors import TNFRValueError
from ._argument_validation import finite_node_real, finite_real

_OPERATOR = "Self-organization"


@dataclass(frozen=True, slots=True)
class TholPressureProposal:
    """Same-call validated scalar pressure write and acceleration telemetry."""

    d2_epi: float
    dnfr_after: float


def propose_thol_pressure(dnfr, d2_epi, gain) -> TholPressureProposal:
    """Materialize p + gain * acceleration before any nodal write."""
    pressure = finite_real(dnfr, operator=_OPERATOR, label="THOL DeltaNFR state")
    acceleration = finite_real(d2_epi, operator=_OPERATOR, label="THOL d2EPI state")
    factor = finite_real(
        gain, operator=_OPERATOR, label="THOL_accel",
        lower=math.nextafter(0.0, math.inf),
    )
    contribution = finite_real(
        factor * acceleration, operator=_OPERATOR, label="THOL DeltaNFR contribution",
    )
    result = finite_real(
        pressure + contribution, operator=_OPERATOR, label="THOL DeltaNFR proposal",
    )
    return TholPressureProposal(acceleration, result)


def prepare_graph_thol_pressure(graph, node, gain) -> TholPressureProposal:
    """Read active physical/legacy history; cached curvature is not evidence."""
    from .nodal_equation import compute_d2epi_dt2
    from .preconditions import OperatorPreconditionError

    try:
        pressure = finite_node_real(
            graph.nodes[node], ALIAS_DNFR, 0.0,
            operator=_OPERATOR, label="THOL DeltaNFR state",
        )
        acceleration = compute_d2epi_dt2(graph, node, store=False)
        return propose_thol_pressure(pressure, acceleration, gain)
    except OperatorPreconditionError as exc:
        # Keep the primitive dispatcher's numerical-error interface.
        raise TNFRValueError(exc.reason) from exc


def commit_graph_thol_pressure(node, proposal: TholPressureProposal) -> None:
    """Commit a graph-backed primitive proposal without creating children."""
    node.dnfr = proposal.dnfr_after
    node.d2EPI = proposal.d2_epi
