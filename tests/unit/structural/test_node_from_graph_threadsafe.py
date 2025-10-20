"""Pruebas de `NodeNX.from_graph` en entornos multihilo."""

from concurrent.futures import ThreadPoolExecutor
import networkx as nx

from tnfr.node import NodeNX


def test_from_graph_thread_safety():
    G = nx.Graph()
    G.add_node(1)
    calls = 0
    original_init = NodeNX.__init__

    def counting_init(self, graph, n):
        nonlocal calls
        calls += 1
        original_init(self, graph, n)

    try:
        NodeNX.__init__ = counting_init
        with ThreadPoolExecutor(max_workers=16) as ex:
            results = list(ex.map(lambda _: NodeNX.from_graph(G, 1), range(32)))
    finally:
        NodeNX.__init__ = original_init

    first = results[0]
    assert all(r is first for r in results)
    assert calls == 1
