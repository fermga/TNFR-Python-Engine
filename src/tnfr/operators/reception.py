"""Reception (EN) operator.

Purpose: integrate incoming neighbour EPI without writing stored ΔNFR.
Physics: active reorganization of local EPI from the neighbour field.
Grammar: follows generator flows; typically EN->IL or EN->THOL.
Telemetry: source list, EPI before/after, and observed stored ΔNFR.
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar

from ..config.operator_names import RECEPTION
from ..types import Glyph, TNFRGraph
from ._reception_kernel import (
    ReceptionReadSnapshot,
    capture_reception_read_snapshot,
    reception_no_sources_warning,
)
from .definitions_base import Operator, _PREPARED_OPERATOR_STATE_KEY


class Reception(Operator):
    """Blend local and neighbour EPI and update the semantic EPI kind.

    Typical: AL->EN->IL, RA->EN, EN->THOL, EN->UM. Avoid EN with SHA.
    Metrics: EPI delta, observed stored ΔNFR and source count.
    """

    __slots__ = ()
    name: ClassVar[str] = RECEPTION
    glyph: ClassVar[Glyph] = Glyph.EN

    def _validate_application_preconditions(
        self,
        G: TNFRGraph,
        node: Any,
        **kw: Any,
    ) -> None:
        """Validate EN execution controls before source reads or warnings."""

        super()._validate_application_preconditions(G, node, **kw)
        from ._argument_validation import nonnegative_integer, strict_bool

        if "track_sources" in kw:
            strict_bool(
                kw["track_sources"],
                operator=self.name,
                label="track_sources",
            )
        if "max_distance" in kw:
            nonnegative_integer(
                kw["max_distance"],
                operator=self.name,
                label="max_distance",
            )

    def _prepare_glyph_application(
        self,
        G: TNFRGraph,
        node: Any,
        **kw: Any,
    ) -> ReceptionReadSnapshot:
        """Freeze source and neighbour inputs at the same pre-EN boundary."""

        track_sources = bool(kw.get("track_sources", True))
        snapshot = capture_reception_read_snapshot(
            G,
            node,
            track_sources=track_sources,
            max_distance=kw.get("max_distance", 2),
        )
        # Direct EN has no outer graph transaction. Keep this advisory before
        # the write so warning-as-error cannot leave a partially applied node.
        # Two-phase stages emit the same text after final evidence validation.
        if track_sources and not snapshot.reception_sources:
            warnings.warn(
                reception_no_sources_warning(node),
                stacklevel=3,
            )
        return snapshot

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Run EN policy validation for EPI, stored pressure and topology."""
        from .preconditions.reception import validate_reception_strict

        validate_reception_strict(G, node)

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect EN metrics (epi before)."""
        from .metrics_basic import _reception_metrics_from_snapshot

        snapshot = state_before.get(_PREPARED_OPERATOR_STATE_KEY)
        if type(snapshot) is not ReceptionReadSnapshot:
            from .metrics import reception_metrics

            return reception_metrics(G, node, state_before["epi"])
        return _reception_metrics_from_snapshot(
            G,
            node,
            state_before["epi"],
            read_snapshot=snapshot,
        )
