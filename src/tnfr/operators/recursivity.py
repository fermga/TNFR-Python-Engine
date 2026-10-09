"""Recursivity (REMESH) operator.

Purpose: request network-scale fractal echo handling across nested EPIs.
Physics: explicit REMESH mixing references epi(t-tau); the glyph is advisory.
Grammar: generator/closure; depth>1 enforces U5 stabilizers nearby.
Telemetry: depth, epi_before, vf_before for recursion analysis.
Typical: THOL->REMESH, REMESH->IL, VAL->REMESH, REMESH->RA.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..config.operator_names import RECURSIVITY
from ..errors import TNFRValueError
from ..types import Glyph, TNFRGraph
from .definitions_base import Operator


def validate_recursivity_depth(depth: Any) -> int:
    """Validate the positive integer scale declaration used by batch U5."""
    from ..validation.window import validate_window

    try:
        return validate_window(depth, positive=True)
    except (TypeError, ValueError) as exc:
        raise TNFRValueError(
            f"depth must be a positive integer, got {depth!r}",
            context={"depth": depth},
        ) from exc


class Recursivity(Operator):
    """Record a network REMESH advisory with declared U5 depth.

    depth>1 requires nearby IL/THOL. Explicit delayed EPI mixing is separate.
    """

    __slots__ = ("depth",)
    name: ClassVar[str] = RECURSIVITY
    glyph: ClassVar[Glyph] = Glyph.REMESH

    def __init__(self, depth: int = 1):
        """Declare recursion depth (positive integer) for batch U5 validation."""
        self.depth = validate_recursivity_depth(depth)

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Run REMESH precondition validator."""
        from .preconditions import validate_recursivity

        validate_recursivity(G, node)

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect advisory REMESH metrics from the unchanged target state."""
        from .metrics import recursivity_metrics

        return recursivity_metrics(G, node, state_before["epi"], state_before["vf"])


_CANONICAL_RECURSIVITY_EXECUTE = Recursivity._execute
