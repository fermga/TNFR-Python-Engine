"""TNFR Operator: Recursivity

Recursivity structural operator (REMESH) - Fractal pattern propagation.

**Physics**: See AGENTS.md § Recursivity
**Grammar**: UNIFIED_GRAMMAR_RULES.md
"""

from __future__ import annotations

import warnings
from typing import Any, ClassVar

from ..alias import get_attr
from ..config.operator_names import RECURSIVITY
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI
from ..types import Glyph, TNFRGraph
from .definitions_base import Operator


class Recursivity(Operator):
    """Recursivity structural operator (REMESH) - Fractal pattern propagation.

    Activates glyph ``REMESH`` to propagate fractal recursivity and echo structural
    patterns across nested EPIs, maintaining multi-scale identity.

    TNFR Context: Recursivity (REMESH) implements operational fractality - patterns that
    replicate across scales while preserving structural identity. REMESH ensures that
    EPI(t) echoes EPI(t - τ) at nested levels, creating self-similar coherence structures.

    Use Cases: Fractal processes, multi-scale coherence, memory recursion, pattern
    replication, self-similar organization, adaptive memory systems.

    Typical Sequences: REMESH → RA (recursive propagation), THOL → REMESH (emergence
    with fractal structure), REMESH → IL (recursive pattern stabilization), VAL → REMESH
    (expansion with self-similarity).

    Critical: REMESH preserves identity across scales - fundamental to TNFR fractality.

    Parameters
    ----------
    depth : int, optional
        Hierarchical nesting depth for multi-scale recursion (default: 1).
        - depth=1: Shallow recursion (single level, no multi-scale constraint)
        - depth>1: Deep recursion (multi-level hierarchy, requires U5 stabilizers)

    Notes
    -----
    **U5: Multi-Scale Coherence**: When depth>1, U5 grammar rule applies requiring
    scale stabilizers (IL or THOL) within ±3 operators to preserve coherence across
    hierarchical levels. This ensures C_parent ≥ α·ΣC_child per conservation principle.

    See UNIFIED_GRAMMAR_RULES.md § U5 for complete physical derivation.

    Examples
    --------
    >>> from tnfr.constants import EPI_PRIMARY, VF_PRIMARY
    >>> from tnfr.dynamics import set_delta_nfr_hook
    >>> from tnfr.structural import create_nfr, run_sequence
    >>> from tnfr.operators.definitions import Recursivity
    >>> G, node = create_nfr("nu", epi=0.52, vf=0.92)
    >>> echoes = iter([(0.02, 0.03)])
    >>> def echo(graph):
    ...     d_epi, d_vf = next(echoes)
    ...     graph.nodes[node][EPI_PRIMARY] += d_epi
    ...     graph.nodes[node][VF_PRIMARY] += d_vf
    ...     graph.graph.setdefault("echo_trace", []).append(
    ...         (round(graph.nodes[node][EPI_PRIMARY], 2), round(graph.nodes[node][VF_PRIMARY], 2))
    ...     )
    >>> set_delta_nfr_hook(G, echo)
    >>> run_sequence(G, node, [Recursivity()])
    >>> G.graph["echo_trace"]
    [(0.54, 0.95)]

    Deep recursion example requiring U5 stabilizers:
    >>> from tnfr.operators.definitions import Recursivity, Coherence, Silence
    >>> # depth=3 creates multi-level hierarchy - requires IL for U5
    >>> ops = [Recursivity(depth=3), Coherence(), Silence()]

    **Biomedical**: Fractal physiology (HRV, EEG), developmental recapitulation
    **Cognitive**: Recursive thinking, meta-cognition, self-referential processes
    **Social**: Cultural fractals, organizational self-similarity, meme propagation
    """

    __slots__ = ("depth",)
    name: ClassVar[str] = RECURSIVITY
    glyph: ClassVar[Glyph] = Glyph.REMESH

    def __init__(self, depth: int = 1):
        """Initialize Recursivity operator with hierarchical depth.

        Parameters
        ----------
        depth : int, optional
            Nesting depth for multi-scale recursion (default: 1)
        """
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self.depth = depth

    def _validate_preconditions(self, G: TNFRGraph, node: Any) -> None:
        """Validate REMESH-specific preconditions."""
        from .preconditions import validate_recursivity

        validate_recursivity(G, node)

    def _collect_metrics(
        self, G: TNFRGraph, node: Any, state_before: dict[str, Any]
    ) -> dict[str, Any]:
        """Collect REMESH-specific metrics."""
        from .metrics import recursivity_metrics

        return recursivity_metrics(G, node, state_before["epi"], state_before["vf"])
