"""TNFR Operator: Expansion

Expansion structural operator (VAL) - Structural dilation for exploration.

**Physics**: See AGENTS.md § Expansion
**Grammar**: UNIFIED_GRAMMAR_RULES.md
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar

from ..alias import get_attr
from ..config.operator_names import EXPANSION
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI
from ..types import Glyph, TNFRGraph
from .definitions_base import Operator
from .registry import register_operator


class Expansion(Operator):
    """Expansion structural operator (VAL) - Structural dilation for exploration.

    Activates glyph ``VAL`` to dilate the node's structure, unfolding neighbouring
    trajectories and extending operational boundaries to explore additional coherence volume.

    TNFR Context: Expansion increases EPI magnitude and νf, enabling exploration of new
    structural configurations while maintaining core identity. VAL embodies fractality -
    structures scale while preserving their essential form.

    Use Cases: Growth processes (biological, cognitive, organizational), exploration phases,
    capacity building, network extension.

    Typical Sequences: VAL → IL (expand then stabilize), OZ → VAL (dissonance enables
    expansion), VAL → THOL (expansion triggers reorganization).

    Avoid: VAL → NUL (contradictory), multiple consecutive VAL without consolidation.

    Examples
    --------
    >>> from tnfr.constants import EPI_PRIMARY, VF_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Expansion
    >>> G, node = create_nfr("theta", epi=0.47, vf=0.95)
    >>> spreads = iter([(0.06, 0.08)])
    >>> def open_volume(graph):
    ...     d_epi, d_vf = next(spreads)
    ...     graph.nodes[node][EPI_PRIMARY] += d_epi
    ...     graph.nodes[node][VF_PRIMARY] += d_vf
    >>> set_delta_nfr_hook(G, open_volume)
    >>> run_sequence(G, node, [Expansion()])
    >>> round(G.nodes[node][EPI_PRIMARY], 2)
    0.53
    >>> round(G.nodes[node][VF_PRIMARY], 2)
    1.03

    **Biomedical**: Growth, tissue expansion, neural network development
    **Cognitive**: Knowledge domain expansion, conceptual broadening
    **Social**: Team scaling, market expansion, network growth
    """

    __slots__ = ()
    name: ClassVar[str] = EXPANSION
    glyph: ClassVar[Glyph] = Glyph.VAL

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Validate VAL-specific preconditions."""
        from .preconditions import validate_expansion

        validate_expansion(G, node)

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect VAL-specific metrics."""
        from .metrics import expansion_metrics

        return expansion_metrics(G, node, state_before["vf"], state_before["epi"])