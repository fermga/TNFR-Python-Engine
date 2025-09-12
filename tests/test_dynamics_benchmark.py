"""Pruebas de dynamics benchmark."""

import time
import networkx as nx
import pytest

from tnfr.dynamics import default_compute_delta_nfr
from tnfr.constants import get_aliases

ALIAS_THETA = get_aliases("THETA")
ALIAS_EPI = get_aliases("EPI")
ALIAS_VF = get_aliases("VF")


@pytest.mark.slow
def test_default_compute_delta_nfr_performance():
    """Simple benchmark to ensure the optimized computation runs quickly."""
    G = nx.gnp_random_graph(200, 0.1, seed=1)
    for n in G.nodes:
        G.nodes[n][ALIAS_THETA] = 0.0
        G.nodes[n][ALIAS_EPI] = 0.0
        G.nodes[n][ALIAS_VF] = 0.0
    start = time.perf_counter()
    default_compute_delta_nfr(G)
    duration = time.perf_counter() - start
    # Should be fast enough on a moderate graph
    assert duration < 1.0
