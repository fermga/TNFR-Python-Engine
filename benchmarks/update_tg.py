"""Benchmark for _update_tg performance."""

import time
import networkx as nx
from collections import Counter, defaultdict

from tnfr.constants import inject_defaults
from tnfr.glyph_history import last_glyph
from tnfr.metrics import _update_tg, _tg_state
from tnfr.metrics.core import LATENT_GLYPH


def _update_tg_naive(G, hist, dt, save_by_node):
    """Reference implementation used for comparison."""
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
        if st.curr is None:
            st.curr = g
            st.run = dt
        elif g == st.curr:
            st.run += dt
        else:
            prev = st.curr
            dur = st.run
            tg_total[prev] += dur
            if save_by_node:
                tg_by_node[n][prev].append(dur)
            st.curr = g
            st.run = dt

    return counts, n_total, n_latent


def _build_graph():
    G = nx.Graph()
    inject_defaults(G)
    G.add_node(0, EPI_kind="OZ")
    G.add_node(1, EPI_kind=LATENT_GLYPH)
    G.add_node(2, EPI_kind="NAV")
    G.add_node(3, EPI_kind="OZ")
    G.add_node(4, EPI_kind=LATENT_GLYPH)
    return G


def run():
    G_opt = _build_graph()
    G_ref = _build_graph()
    hist_opt = {}
    hist_ref = {}
    dt = 1.0

    start = time.perf_counter()
    _update_tg(G_opt, hist_opt, dt, True)
    t_opt = time.perf_counter() - start

    start = time.perf_counter()
    _update_tg_naive(G_ref, hist_ref, dt, True)
    t_ref = time.perf_counter() - start

    print(f"optimized: {t_opt:.6f}s, naive: {t_ref:.6f}s")


if __name__ == "__main__":
    run()
