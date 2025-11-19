"""Reception (EN) operator.

Purpose: integrate external coherence; reduce ΔNFR via structured intake.
Physics: active reorganization of local EPI from network resonance.
Grammar: follows generator flows; typically EN->IL or EN->THOL.
Telemetry: source list, epi_before, epi_after, dnfr shift.
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar

from ..config.operator_names import RECEPTION
from ..types import Glyph, TNFRGraph
from .definitions_base import Operator
 

class Reception(Operator):
    """Integrate external resonance; reduce ΔNFR; preserve pattern identity.

    Typical: AL->EN->IL, RA->EN, EN->THOL, EN->UM. Avoid EN with SHA.
    Metrics: epi delta, dnfr delta, sources count.
    """

    __slots__ = ()
    name: ClassVar[str] = RECEPTION
    glyph: ClassVar[Glyph] = Glyph.EN

    def __call__(self, G: TNFRGraph, node: Any, **kw: Any) -> None:
        """Detect sources (optional); apply grammar; integrate intake."""
        # Detect emission sources BEFORE applying reception
        if kw.get("track_sources", True):
            from .network_analysis.source_detection import (
                detect_emission_sources,
            )

            max_distance = kw.get("max_distance", 2)
            sources = detect_emission_sources(
                G, node, max_distance=max_distance
            )

            # Store detected sources in node metadata for metrics and analysis
            G.nodes[node]["_reception_sources"] = sources

            # Warn if no compatible sources found
            if not sources:
                warnings.warn(
                    (
                        f"EN: node {node} has no sources; "
                        f"external coherence not integrated."
                    ),
                    stacklevel=2,
                )

        # Delegate to parent __call__ which applies grammar
        super().__call__(G, node, **kw)

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Run EN precondition validation (capacity, dnfr, sources)."""
        from .preconditions.reception import validate_reception_strict

        validate_reception_strict(G, node)

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect EN metrics (epi before)."""
        from .metrics import reception_metrics

        return reception_metrics(G, node, state_before["epi"])
