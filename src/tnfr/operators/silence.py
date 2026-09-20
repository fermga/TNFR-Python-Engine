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

from ..config.operator_names import SILENCE
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
        """Validate the shared SHA proposal before recording latency."""
        from datetime import datetime, timezone

        from .al_sha_stage_proposals import (
            commit_silence_lifecycle,
            propose_silence_stage,
        )
        from .factor_contracts import resolve_runtime_operator_factors

        factors = resolve_runtime_operator_factors(
            G.graph.get("GLYPH_FACTORS"), self.glyph, G.graph
        )
        proposal = propose_silence_stage(
            G, node, factors, timestamp=datetime.now(timezone.utc).isoformat()
        )
        commit_silence_lifecycle(G, proposal)
        super()._execute(G, node, **kw)

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
