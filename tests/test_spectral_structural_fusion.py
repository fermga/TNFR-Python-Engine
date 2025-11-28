import types

import networkx as nx
import numpy as np

from tnfr.dynamics.spectral_structural_fusion import (
    TNFRSpectralStructuralFusionEngine,
)
from tnfr.dynamics.structural_cache import get_structural_cache


def _build_graph(num_nodes: int = 5) -> nx.Graph:
    graph = nx.path_graph(num_nodes)
    for node in graph.nodes():
        graph.nodes[node]["EPI"] = float(node)
        graph.nodes[node]["nu_f"] = 1.0 + 0.05 * node
        graph.nodes[node]["phase"] = np.pi * node / max(1, num_nodes - 1)
    return graph


def test_fusion_engine_persists_spectral_signature() -> None:
    engine = TNFRSpectralStructuralFusionEngine()
    graph = _build_graph()

    entry = engine.compute_structural_fields(graph)
    assert entry.spectral_basis_signature

    cached_signature = engine.get_cached_spectral_signature(graph)
    assert cached_signature == entry.spectral_basis_signature


def test_fusion_engine_registers_coordination_nodes() -> None:
    engine = TNFRSpectralStructuralFusionEngine()
    graph = _build_graph()

    coordination_node = types.SimpleNamespace(node_id=0)
    engine.coordinate_cache_with_central_nodes(graph, [coordination_node])

    cache = get_structural_cache()
    entry = cache.get_structural_fields(graph)
    assert 0 in entry.coordination_nodes
