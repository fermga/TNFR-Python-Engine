import networkx as nx

from tnfr.constants import attach_defaults
from tnfr.metrics import _metrics_step


def test_pp_val_zero_when_no_remesh():
    """PP metric should be 0.0 when no RE’MESH events occur."""
    G = nx.Graph()
    attach_defaults(G)
    # Nodo en estado SH’A, pero sin eventos RE’MESH
    G.add_node(0, EPI_kind="SH’A")

    _metrics_step(G)

    morph = G.graph["history"]["morph"][0]
    assert morph["PP"] == 0.0
