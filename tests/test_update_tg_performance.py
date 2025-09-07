"""Pruebas de update tg performance."""

import time
from collections import Counter, defaultdict

from tnfr.constants import attach_defaults
from tnfr.glyph_history import last_glyph
from tnfr.metrics import _update_tg, _tg_state
from tnfr.metrics.core import LATENT_GLYPH, TgCurr, TgRun


def _update_tg_naive(G, hist, dt, save_by_node):
    """Referencia ingenua para comparar resultados con _update_tg."""
    counts = Counter()
    n_total = 0
    n_latent = 0

    tg_total = hist.setdefault("Tg_total", defaultdict(float))
    tg_by_node = (
        hist.setdefault("Tg_by_node", defaultdict(lambda: defaultdict(list)))
        if save_by_node
        else None
    )

    for n in G.nodes():
        nd = G.nodes[n]
        g = last_glyph(nd)
        if not g:
            continue

        n_total += 1
        if g == LATENT_GLYPH:
            n_latent += 1

        counts[g] += 1

        st = _tg_state(nd)
        if st[TgCurr] is None:
            st[TgCurr] = g
            st[TgRun] = dt
        elif g == st[TgCurr]:
            st[TgRun] += dt
        else:
            prev = st[TgCurr]
            dur = float(st[TgRun])
            tg_total[prev] += dur
            if save_by_node:
                tg_by_node[n][prev].append(dur)
            st[TgCurr] = g
            st[TgRun] = dt

    return counts, n_total, n_latent


def test_update_tg_matches_naive(graph_canon):
    """_update_tg produce los mismos resultados que la versión ingenua."""
    G_opt = graph_canon()
    G_ref = graph_canon()

    for G in (G_opt, G_ref):
        G.add_node(0, EPI_kind="OZ")
        G.add_node(1, EPI_kind=LATENT_GLYPH)
        G.add_node(2, EPI_kind="NAV")
        G.add_node(3, EPI_kind="OZ")
        G.add_node(4, EPI_kind=LATENT_GLYPH)
        attach_defaults(G)

    hist_opt = {}
    hist_ref = {}
    dt = 1.0

    start = time.perf_counter()
    counts_opt, n_total_opt, n_latent_opt = _update_tg(
        G_opt, hist_opt, dt, True
    )
    t_opt = time.perf_counter() - start

    start = time.perf_counter()
    counts_ref, n_total_ref, n_latent_ref = _update_tg_naive(
        G_ref, hist_ref, dt, True
    )
    t_ref = time.perf_counter() - start

    assert counts_opt == counts_ref
    assert n_total_opt == n_total_ref
    assert n_latent_opt == n_latent_ref
    assert hist_opt == hist_ref
    assert t_opt <= t_ref * 3
