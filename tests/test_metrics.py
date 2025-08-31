import networkx as nx

from tnfr.constants import attach_defaults
from tnfr.metrics import _metrics_step


def test_pp_val_zero_when_no_remesh():
    """PP metric should be 0.0 when no REMESH events occur."""
    G = nx.Graph()
    attach_defaults(G)
    # Nodo en estado SHA, pero sin eventos REMESH
    G.add_node(0, EPI_kind="SHA")

    _metrics_step(G)

    morph = G.graph["history"]["morph"][0]
    assert morph["PP"] == 0.0
