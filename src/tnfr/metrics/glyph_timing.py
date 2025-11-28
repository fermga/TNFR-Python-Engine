"""Glyph timing utilities and advanced metrics."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from types import ModuleType
from typing import (
    Any,
    Callable,
    Mapping,
    MutableMapping,
    Sequence,
    cast,
)

from ..alias import get_attr
from ..config.constants import GLYPH_GROUPS, GLYPHS_CANONICAL
from ..constants import get_param
from ..constants.aliases import ALIAS_EPI
from ..glyph_history import append_metric
from ..glyph_runtime import last_glyph
from ..utils import resolve_chunk_size
from ..types import (
    GlyphCounts,
    GlyphMetricsHistory,
    GlyphTimingByNode,
    GlyphTimingTotals,
    GlyphogramRow,
    GraphLike,
    MetricsListHistory,
    SigmaTrace,
)

LATENT_GLYPH: str = "SHA"
DEFAULT_EPI_SUPPORT_LIMIT = 0.05

try:  # pragma: no cover - import guard exercised via tests
    import numpy as _np  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - numpy optional dependency
    _np = None

np: ModuleType | None = cast(ModuleType | None, _np)


def _has_numpy_support(np_obj: object) -> bool:
    """Return ``True`` when ``np_obj`` exposes the required NumPy API."""

    return isinstance(np_obj, ModuleType) or (
        np_obj is not None and hasattr(np_obj, "fromiter") and hasattr(np_obj, "bincount")
    )


_GLYPH_TO_INDEX = {glyph: idx for idx, glyph in enumerate(GLYPHS_CANONICAL)}


def _coerce_float(value: Any) -> float:
    """Attempt to coerce ``value`` to ``float`` returning ``0.0`` on failure."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


@dataclass
class GlyphTiming:
    """Mutable accumulator tracking the active glyph and its dwell time."""

    curr: str | None = None
    run: float = 0.0


__all__ = [
    "LATENT_GLYPH",
    "GlyphTiming",
    "SigmaTrace",
    "GlyphogramRow",
    "GlyphTimingTotals",
    "GlyphTimingByNode",
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


def _count_glyphs_chunk(chunk: Sequence[str]) -> Counter[str]:
    """Count glyph occurrences within a chunk (multiprocessing helper)."""

    counter: Counter[str] = Counter()
    for glyph in chunk:
        counter[glyph] += 1
    return counter


def _epi_support_chunk(values: Sequence[float], threshold: float) -> tuple[float, int]:
    """Compute EPI support contribution for a chunk."""

    total = 0.0
    count = 0
    for value in values:
        if value >= threshold:
            total += value
            count += 1
    return total, count


def _tg_state(nd: MutableMapping[str, Any]) -> GlyphTiming:
    """Expose per-node glyph timing state."""

    return nd.setdefault("_Tg", GlyphTiming())


def for_each_glyph(fn: Callable[[str], None]) -> None:
    """Apply ``fn`` to each canonical structural operator."""

    for g in GLYPHS_CANONICAL:
        fn(g)


# ---------------------------------------------------------------------------
# Glyph timing helpers
# ---------------------------------------------------------------------------


def _update_tg_node(
    n: Any,
    nd: MutableMapping[str, Any],
    dt: float,
    tg_total: GlyphTimingTotals,
    tg_by_node: GlyphTimingByNode | None,
) -> tuple[str | None, bool]:
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


def _update_tg(
    G: GraphLike,
    hist: GlyphMetricsHistory,
    dt: float,
    save_by_node: bool,
    n_jobs: int | None = None,
) -> tuple[Counter[str], int, int]:
    """Accumulate glyph dwell times for the entire graph."""

    tg_total = cast(GlyphTimingTotals, hist.setdefault("Tg_total", defaultdict(float)))
    tg_by_node = (
        cast(
            GlyphTimingByNode,
            hist.setdefault(
                "Tg_by_node",
                defaultdict(lambda: defaultdict(list)),
            ),
        )
        if save_by_node
        else None
    )

    n_total = 0
    n_latent = 0
    glyph_sequence: list[str] = []
    for n, nd in G.nodes(data=True):
        g, is_latent = _update_tg_node(n, nd, dt, tg_total, tg_by_node)
        if g is None:
            continue
        n_total += 1
        if is_latent:
            n_latent += 1
        glyph_sequence.append(g)

    counts: Counter[str] = Counter()
    if not glyph_sequence:
        return counts, n_total, n_latent

    if _has_numpy_support(np):
        glyph_idx = np.fromiter(
            (_GLYPH_TO_INDEX[glyph] for glyph in glyph_sequence),
            dtype=np.int64,
            count=len(glyph_sequence),
        )
        freq = np.bincount(glyph_idx, minlength=len(GLYPHS_CANONICAL))
        counts.update(
            {
                glyph: int(freq[_GLYPH_TO_INDEX[glyph]])
                for glyph in GLYPHS_CANONICAL
                if freq[_GLYPH_TO_INDEX[glyph]]
            }
        )
    elif n_jobs is not None and n_jobs > 1 and len(glyph_sequence) > 1:
        approx_chunk = math.ceil(len(glyph_sequence) / n_jobs) if n_jobs else None
        chunk_size = resolve_chunk_size(
            approx_chunk,
            len(glyph_sequence),
            minimum=1,
        )
        futures = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for start in range(0, len(glyph_sequence), chunk_size):
                chunk = glyph_sequence[start : start + chunk_size]
                futures.append(executor.submit(_count_glyphs_chunk, chunk))
            for future in futures:
                counts.update(future.result())
    else:
        counts.update(glyph_sequence)

    return counts, n_total, n_latent


def _update_glyphogram(
    G: GraphLike,
    hist: GlyphMetricsHistory,
    counts: GlyphCounts,
    t: float,
    n_total: int,
) -> None:
    """Record glyphogram row from glyph counts."""

    normalize_series = bool(get_param(G, "METRICS").get("normalize_series", False))
    row: GlyphogramRow = {"t": t}
    total = max(1, n_total)
    for g in GLYPHS_CANONICAL:
        c = counts.get(g, 0)
        row[g] = (c / total) if normalize_series else c
    append_metric(cast(MetricsListHistory, hist), "glyphogram", row)


def _update_latency_index(
    G: GraphLike,
    hist: GlyphMetricsHistory,
    n_total: int,
    n_latent: int,
    t: float,
) -> None:
    """Record latency index for the current step."""

    li = n_latent / max(1, n_total)
    append_metric(
        cast(MetricsListHistory, hist),
        "latency_index",
        {"t": t, "value": li},
    )


def _update_epi_support(
    G: GraphLike,
    hist: GlyphMetricsHistory,
    t: float,
    threshold: float = DEFAULT_EPI_SUPPORT_LIMIT,
    n_jobs: int | None = None,
) -> None:
    """Measure EPI support and normalized magnitude."""

    node_count = G.number_of_nodes()
    total = 0.0
    count = 0

    if _has_numpy_support(np) and node_count:
        epi_values = np.fromiter(
            (abs(_coerce_float(get_attr(nd, ALIAS_EPI, 0.0))) for _, nd in G.nodes(data=True)),
            dtype=float,
            count=node_count,
        )
        mask = epi_values >= threshold
        count = int(mask.sum())
        if count:
            total = float(epi_values[mask].sum())
    elif n_jobs is not None and n_jobs > 1 and node_count > 1:
        values = [abs(_coerce_float(get_attr(nd, ALIAS_EPI, 0.0))) for _, nd in G.nodes(data=True)]
        approx_chunk = math.ceil(len(values) / n_jobs) if n_jobs else None
        chunk_size = resolve_chunk_size(
            approx_chunk,
            len(values),
            minimum=1,
        )
        totals: list[tuple[float, int]] = []
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = []
            for start in range(0, len(values), chunk_size):
                chunk = values[start : start + chunk_size]
                futures.append(executor.submit(_epi_support_chunk, chunk, threshold))
            for future in futures:
                totals.append(future.result())
        for part_total, part_count in totals:
            total += part_total
            count += part_count
    else:
        for _, nd in G.nodes(data=True):
            epi_val = abs(_coerce_float(get_attr(nd, ALIAS_EPI, 0.0)))
            if epi_val >= threshold:
                total += epi_val
                count += 1
    epi_norm = (total / count) if count else 0.0
    append_metric(
        cast(MetricsListHistory, hist),
        "EPI_support",
        {"t": t, "size": count, "epi_norm": float(epi_norm)},
    )


def _update_morph_metrics(
    G: GraphLike,
    hist: GlyphMetricsHistory,
    counts: GlyphCounts,
    t: float,
) -> None:
    """Capture morphosyntactic distribution of glyphs."""

    def get_count(keys: Sequence[str]) -> int:
        return sum(counts.get(k, 0) for k in keys)

    total = max(1, sum(counts.values()))
    id_val = get_count(GLYPH_GROUPS.get("ID", ())) / total
    cm_val = get_count(GLYPH_GROUPS.get("CM", ())) / total
    ne_val = get_count(GLYPH_GROUPS.get("NE", ())) / total
    num = get_count(GLYPH_GROUPS.get("PP_num", ()))
    den = get_count(GLYPH_GROUPS.get("PP_den", ()))
    pp_val = 0.0 if den == 0 else num / den
    append_metric(
        cast(MetricsListHistory, hist),
        "morph",
        {"t": t, "ID": id_val, "CM": cm_val, "NE": ne_val, "PP": pp_val},
    )


def _compute_advanced_metrics(
    G: GraphLike,
    hist: GlyphMetricsHistory,
    t: float,
    dt: float,
    cfg: Mapping[str, Any],
    threshold: float = DEFAULT_EPI_SUPPORT_LIMIT,
    n_jobs: int | None = None,
) -> None:
    """Compute glyph timing derived metrics."""

    save_by_node = bool(cfg.get("save_by_node", True))
    counts, n_total, n_latent = _update_tg(G, hist, dt, save_by_node, n_jobs=n_jobs)
    _update_glyphogram(G, hist, counts, t, n_total)
    _update_latency_index(G, hist, n_total, n_latent, t)
    _update_epi_support(G, hist, t, threshold, n_jobs=n_jobs)
    _update_morph_metrics(G, hist, counts, t)
