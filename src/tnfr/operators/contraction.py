"""Contraction (NUL) operator.

Purpose: reduce vf and reciprocally scale stored delta NFR.
Grammar: support; sequence admission is separate from a stability theorem.
Typical: VAL->NUL->IL; THOL->NUL (refine emergent structure).
Avoid: chain NUL; NUL->OZ (destabilize); NUL when EPI≈0.
Effects: optional edge-aware EPI scaling; phase and support unchanged.
Preconditions: non-trivial EPI; integrity; recent expansion optional.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..config.operator_names import CONTRACTION
from ..types import Glyph, TNFRGraph
from .definitions_base import Operator


class Contraction(Operator):
    """Attenuate capacity and multiply stored pressure by its reciprocal gain.

    The ideal stored nodal product is unchanged. The default edge-aware
    branch also scales/projects EPI; disabling it preserves EPI. Neither
    graph dimension nor physical volume is defined by this map, and a later
    pressure refresh need not preserve the stored-product identity.
    """

    __slots__ = ()
    name: ClassVar[str] = CONTRACTION
    glyph: ClassVar[Glyph] = Glyph.NUL

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        from .preconditions import validate_contraction

        validate_contraction(G, node)

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        from .metrics import contraction_metrics

        return contraction_metrics(
            G,
            node,
            state_before["vf"],
            state_before["epi"],
            dnfr_before=state_before["dnfr"],
        )


__all__ = ["Contraction"]
