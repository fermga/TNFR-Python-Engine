"""Glyph timing utilities and advanced metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable

from ..alias import get_attr
from ..constants import get_aliases, get_param
from ..constants_glyphs import GLYPH_GROUPS, GLYPHS_CANONICAL
from ..glyph_history import append_metric, last_glyph
from ..types import Glyph

ALIAS_EPI = get_aliases("EPI")

LATENT_GLYPH = Glyph.SHA.value


@dataclass
class GlyphTiming:
    curr: str | None = None
    run: float = 0.0


__all__ = [
    "LATENT_GLYPH",
    "GlyphTiming",
    "_tg_state",
    "for_each_glyph",
    "_update_tg_node",
    "_update_tg",
    "_update_glyphogram",
    "_update_latency_index",
    "_update_epi_support",
    "_update_morph_metrics",
    "_compute_advanced_metrics",
]


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _tg_state(nd: dict[str, Any]) -> GlyphTiming:
    """Expose per-node glyph timing state."""

    return nd.setdefault("_Tg", GlyphTiming())


def for_each_glyph(fn: Callable[[str], Any]) -> None:
    """Apply ``fn`` to each canonical structural operator."""

    for g in GLYPHS_CANONICAL:
        fn(g)


# ---------------------------------------------------------------------------
# Glyph timing helpers
# ---------------------------------------------------------------------------


def _update_tg_node(n, nd, dt, tg_total, tg_by_node):
    """Track a node's glyph transition and accumulate run time."""

    g = last_glyph(nd)
    if not g:
        return None, False
    st = _tg_state(nd)
    curr = st.curr
    if curr is None:
        st.curr = g
        st.run = dt
    elif g == curr:
        st.run += dt
    else:
        dur = st.run
        tg_total[curr] += dur
        if tg_by_node is not None:
            tg_by_node[n][curr].append(dur)
        st.curr = g
        st.run = dt
    return g, g == LATENT_GLYPH


def _update_tg(G, hist, dt, save_by_node: bool):
    """Accumulate glyph dwell times for the entire graph."""

    counts = Counter()
    tg_total = hist.setdefault("Tg_total", defaultdict(float))
    tg_by_node = (
        hist.setdefault("Tg_by_node", defaultdict(lambda: defaultdict(list)))
        if save_by_node
        else None
    )

    n_total = 0
    n_latent = 0
    for n, nd in G.nodes(data=True):
        g, is_latent = _update_tg_node(n, nd, dt, tg_total, tg_by_node)
        if g is None:
            continue
        n_total += 1
        if is_latent:
            n_latent += 1
        counts[g] += 1
    return counts, n_total, n_latent


def _update_glyphogram(G, hist, counts, t, n_total):
    """Record glyphogram row from glyph counts."""

    normalize_series = bool(get_param(G, "METRICS").get("normalize_series", False))
    row = {"t": t}
    total = max(1, n_total)
    for g in GLYPHS_CANONICAL:
        c = counts.get(g, 0)
        row[g] = (c / total) if normalize_series else c
    append_metric(hist, "glyphogram", row)


def _update_latency_index(G, hist, n_total, n_latent, t):
    """Record latency index for the current step."""

    li = n_latent / max(1, n_total)
    append_metric(hist, "latency_index", {"t": t, "value": li})


def _update_epi_support(G, hist, t, thr):
    """Measure EPI support and normalized magnitude."""

    total = 0.0
    count = 0
    for _, nd in G.nodes(data=True):
        epi_val = abs(get_attr(nd, ALIAS_EPI, 0.0))
        if epi_val >= thr:
            total += epi_val
            count += 1
    epi_norm = (total / count) if count else 0.0
    append_metric(
        hist,
        "EPI_support",
        {"t": t, "size": count, "epi_norm": float(epi_norm)},
    )


def _update_morph_metrics(G, hist, counts, t):
    """Capture morphosyntactic distribution of glyphs."""

    def get_count(keys):
        return sum(counts.get(k, 0) for k in keys)

    total = max(1, sum(counts.values()))
    id_val = get_count(GLYPH_GROUPS.get("ID", ())) / total
    cm_val = get_count(GLYPH_GROUPS.get("CM", ())) / total
    ne_val = get_count(GLYPH_GROUPS.get("NE", ())) / total
    num = get_count(GLYPH_GROUPS.get("PP_num", ()))
    den = get_count(GLYPH_GROUPS.get("PP_den", ()))
    pp_val = 0.0 if den == 0 else num / den
    append_metric(
        hist,
        "morph",
        {"t": t, "ID": id_val, "CM": cm_val, "NE": ne_val, "PP": pp_val},
    )


def _compute_advanced_metrics(G, hist, t, dt, cfg, thr):
    """Compute glyph timing derived metrics."""

    save_by_node = bool(cfg.get("save_by_node", True))
    counts, n_total, n_latent = _update_tg(G, hist, dt, save_by_node)
    _update_glyphogram(G, hist, counts, t, n_total)
    _update_latency_index(G, hist, n_total, n_latent, t)
    _update_epi_support(G, hist, t, thr)
    _update_morph_metrics(G, hist, counts, t)
