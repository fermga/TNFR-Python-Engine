"""Canonical grammar enforcement utilities."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING, cast

from ..constants import DEFAULTS, get_param
from ..operators import apply_glyph
from ..types import Glyph, NodeId, TNFRGraph
from .compatibility import CANON_COMPAT
from . import rules as _rules

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..node import NodeProtocol

__all__ = [
    "GrammarContext",
    "_gram_state",
    "enforce_canonical_grammar",
    "on_applied_glyph",
    "apply_glyph_with_grammar",
]


@dataclass
class GrammarContext:
    """Shared context for grammar helpers.

    Collects graph-level settings to reduce positional parameters across
    helper functions.
    """

    G: TNFRGraph
    cfg_soft: dict[str, Any]
    cfg_canon: dict[str, Any]
    norms: dict[str, Any]

    @classmethod
    def from_graph(cls, G: TNFRGraph) -> "GrammarContext":
        """Create a :class:`GrammarContext` for ``G``."""

        return cls(
            G=G,
            cfg_soft=G.graph.get("GRAMMAR", DEFAULTS.get("GRAMMAR", {})),
            cfg_canon=G.graph.get(
                "GRAMMAR_CANON", DEFAULTS.get("GRAMMAR_CANON", {})
            ),
            norms=G.graph.get("_sel_norms") or {},
        )


# -------------------------
# Per-node grammar state
# -------------------------

def _gram_state(nd: dict[str, Any]) -> dict[str, Any]:
    """Create or return the node grammar state."""

    return nd.setdefault("_GRAM", {"thol_open": False, "thol_len": 0})


# -------------------------
# Core: enforce grammar on a candidate
# -------------------------

def enforce_canonical_grammar(
    G: TNFRGraph,
    n: NodeId,
    cand: Glyph | str,
    ctx: Optional[GrammarContext] = None,
) -> Glyph | str:
    """Validate and adjust a candidate glyph according to canonical grammar."""

    if ctx is None:
        ctx = GrammarContext.from_graph(G)

    nd = ctx.G.nodes[n]
    st = _gram_state(nd)

    raw_cand = cand
    cand = _rules.coerce_glyph(cand)
    input_was_str = isinstance(raw_cand, str)

    # 0) If glyphs outside the alphabet arrive, leave untouched
    if not isinstance(cand, Glyph) or cand not in CANON_COMPAT:
        return raw_cand if input_was_str else cand

    original = cand
    cand = _rules._check_repeats(ctx, n, cand)

    cand = _rules._maybe_force(ctx, n, cand, original, _rules.normalized_dnfr, "force_dnfr")
    cand = _rules._maybe_force(ctx, n, cand, original, _rules._accel_norm, "force_accel")
    cand = _rules._check_oz_to_zhir(ctx, n, cand)
    cand = _rules._check_thol_closure(ctx, n, cand, st)
    cand = _rules._check_compatibility(ctx, n, cand)

    coerced_final = _rules.coerce_glyph(cand)
    if input_was_str:
        if isinstance(coerced_final, Glyph):
            return coerced_final.value
        return str(cand)
    return coerced_final if isinstance(coerced_final, Glyph) else cand


# -------------------------
# Post-selection: update grammar state
# -------------------------

def on_applied_glyph(G: TNFRGraph, n: NodeId, applied: Glyph | str) -> None:
    nd = G.nodes[n]
    st = _gram_state(nd)
    try:
        glyph = applied if isinstance(applied, Glyph) else Glyph(str(applied))
    except ValueError:
        glyph = None

    if glyph is Glyph.THOL:
        st["thol_open"] = True
        st["thol_len"] = 0
    elif glyph in (Glyph.SHA, Glyph.NUL):
        st["thol_open"] = False
        st["thol_len"] = 0


# -------------------------
# Direct application with canonical grammar
# -------------------------

def apply_glyph_with_grammar(
    G: TNFRGraph,
    nodes: Optional[Iterable[NodeId | "NodeProtocol"]],
    glyph: Glyph | str,
    window: Optional[int] = None,
) -> None:
    """Apply ``glyph`` to ``nodes`` enforcing the canonical grammar."""

    if window is None:
        window = get_param(G, "GLYPH_HYSTERESIS_WINDOW")

    g_str = glyph.value if isinstance(glyph, Glyph) else str(glyph)
    iter_nodes = G.nodes() if nodes is None else nodes
    ctx = GrammarContext.from_graph(G)
    for node_ref in iter_nodes:
        node_id = cast(NodeId, getattr(node_ref, "n", node_ref))
        g_eff = enforce_canonical_grammar(G, node_id, g_str, ctx)
        apply_glyph(G, node_id, g_eff, window=window)
        on_applied_glyph(G, node_id, g_eff)
