"""Silence (SHA) operator.

Purpose: structural pause; preserve epi by lowering vf ~0.
Physics: νf≈0 => dEPI/dt≈0 even if dnfr present.
Grammar: closure (U1b); used after IL for long-term retention.
Effects: epi invariant; vf suppressed; dnfr frozen; theta unchanged.
Preconditions: existing epi; dnfr not critical; context allows inactivity.
Typical: IL->SHA; SHA->IL->AL; OZ->SHA (containment); SHA->NAV.
Avoid: SHA->AL direct; SHA->OZ; redundant SHA->SHA.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..config.operator_names import SILENCE
from ..types import Glyph, TNFRGraph
from .definitions_base import Operator


class Silence(Operator):
    """Lower vf; hold epi invariant; set latency tracking attributes.

    Invariants: epi preserved; vf suppressed; dnfr unchanged; theta stable.
    Typical: IL->SHA; SHA->IL->AL; OZ->SHA containment; SHA->NAV.
    Latency attrs: latent, latency_start_time, preserved_epi, silence_duration.
    """

    __slots__ = ()
    name: ClassVar[str] = SILENCE
    glyph: ClassVar[Glyph] = Glyph.SHA

    def __call__(self, G: TNFRGraph, node: Any, **kw: Any) -> None:
        """Mark latency then apply base operator."""
        # Mark latency state BEFORE grammar execution
        self._mark_latency_state(G, node)

        # Delegate to parent __call__ which applies grammar
        super().__call__(G, node, **kw)

    def _mark_latency_state(self, G: TNFRGraph, node: Any) -> None:
        """Set latent flag, timestamp, preserved epi, duration=0.0."""
        from datetime import datetime, timezone

        G.nodes[node]["latent"] = True
        G.nodes[node]["latency_start_time"] = (
            datetime.now(timezone.utc).isoformat()
        )
        epi_attr = getattr(G.nodes[node], "epi", G.nodes[node].get("epi", 0.0))
        epi_value = float(epi_attr)
        G.nodes[node]["preserved_epi"] = epi_value
        G.nodes[node]["silence_duration"] = 0.0

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Validate SHA-specific preconditions."""
        from .preconditions import validate_silence

        validate_silence(G, node)

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect SHA-specific metrics."""
        from .metrics import silence_metrics

        return silence_metrics(
            G,
            node,
            state_before["vf"],
            state_before["epi"],
        )
