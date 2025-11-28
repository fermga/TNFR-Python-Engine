"""Contraction (NUL) operator.

Purpose: densify & consolidate; reduce vf; amplify local delta NFR.
Grammar: support; pair with IL for safe stabilization.
Typical: VAL->NUL->IL; THOL->NUL (refine emergent structure).
Avoid: chain NUL; NUL->OZ (destabilize); NUL when EPI≈0.
Effects: volume shrinks; delta NFR density up; coherence may tighten.
Preconditions: non-trivial EPI; integrity; recent expansion optional.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..config.operator_names import CONTRACTION
from ..types import Glyph, TNFRGraph
from .definitions_base import Operator


class Contraction(Operator):
    """Densify structure; amplify local delta NFR; prep for IL.

    Minimal contraction consolidates exploration (VAL/THOL). Follow with
    IL or SHA to preserve coherence; avoid chaining or NUL->OZ.
    """

    __slots__ = ()
    name: ClassVar[str] = CONTRACTION
    glyph: ClassVar[Glyph] = Glyph.NUL

    def _validate_preconditions(
        self, G: TNFRGraph, node: Any
    ) -> None:
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
        )


__all__ = ["Contraction"]
