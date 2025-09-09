"""Pruebas de node all nodes."""

import hashlib
import tracemalloc

import networkx as nx

from tnfr.node import NodoTNFR
from tnfr.helpers.cache import node_set_checksum, _node_repr, _hash_node


def test_all_nodes_returns_full_list():
    a = NodoTNFR()
    b = NodoTNFR()
    graph = {"_all_nodes": [a, b]}
    a.graph = graph
    b.graph = graph

    assert set(a.all_nodes()) == {a, b}
    assert set(b.all_nodes()) == {a, b}


def _reference_checksum(G: nx.Graph) -> str:
    reps = sorted(_node_repr(n) for n in G.nodes())
    hasher = hashlib.blake2b(digest_size=16)
    for rep in reps:
        digest = hashlib.blake2b(rep.encode("utf-8"), digest_size=16).digest()
        hasher.update(digest)
    return hasher.hexdigest()


def _peak_memory(fn, *args) -> int:
    _node_repr.cache_clear()
    _hash_node.cache_clear()
    tracemalloc.start()
    fn(*args)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak


def test_node_set_checksum_peak_memory_reduced(graph_canon):
    G = graph_canon()
    for i in range(100_000):
        G.add_node(f"node-{i:05d}" * 2)
    peak_new = _peak_memory(node_set_checksum, G)
    peak_old = _peak_memory(_reference_checksum, G)
    assert peak_new < peak_old
