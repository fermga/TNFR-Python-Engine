"""Silence (SHA) operator.

Purpose: attenuate capacity and mark latency without changing epi.
Physics: later nodal flow vanishes only when νf·ΔNFR=0.
Grammar: closure (U1b); used after IL for structural latency.
Effects at the event: epi, dnfr and theta unchanged; vf attenuated.
Preconditions: existing epi; dnfr not critical; context allows inactivity.
Typical: IL->SHA; SHA->IL->AL; OZ->SHA (containment); SHA->NAV.
Avoid: SHA->AL direct; SHA->OZ; redundant SHA->SHA.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..alias import get_attr
from ..config.operator_names import SILENCE
from ..constants.aliases import ALIAS_EPI
from ..types import Glyph, TNFRGraph
from .definitions_base import Operator


class Silence(Operator):
    """Lower vf; preserve epi during the event; set latency tracking attributes.

    Event invariants: epi, dnfr and theta unchanged; vf attenuated.
    Latency tracking alone does not guarantee zero subsequent nodal flow.
    Typical: IL->SHA; SHA->IL->AL; OZ->SHA containment; SHA->NAV.
    Latency attrs: latent, latency_start_time, preserved_epi, silence_duration.
    """

    __slots__ = ()
    name: ClassVar[str] = SILENCE
    glyph: ClassVar[Glyph] = Glyph.SHA

    def _execute(self, G: TNFRGraph, node: Any, **kw: Any) -> None:
        """Mark latency then apply base operator."""
        # Grammar and preconditions already passed at the public entry point.
        self._mark_latency_state(G, node)

        # Apply the selected glyph with latency metadata ready for metrics.
        super()._execute(G, node, **kw)

    def _mark_latency_state(self, G: TNFRGraph, node: Any) -> None:
        """set latent flag, timestamp, preserved epi, duration=0.0.

        Enhanced for initial nodes: respects TNFR nodal dynamics while
        providing appropriate EPI preservation tracking.
        """
        from datetime import datetime, timezone

        G.nodes[node]["latent"] = True
        G.nodes[node]["latency_start_time"] = datetime.now(timezone.utc).isoformat()
        epi_value = float(get_attr(G.nodes[node], ALIAS_EPI, 0.0))
        G.nodes[node]["preserved_epi"] = epi_value
        G.nodes[node]["silence_duration"] = 0.0

        # Mark initial node status for enhanced tolerance
        G.nodes[node]["was_initial_on_silence"] = abs(epi_value) < 1e-6

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
