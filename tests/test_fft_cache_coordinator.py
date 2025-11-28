import numpy as np
import networkx as nx

from tnfr.dynamics.fft_cache_coordinator import FFTCacheCoordinator
from tnfr.dynamics.advanced_fft_arithmetic import (
    FFTArithmeticResult,
    SpectralOperation,
)


def _make_graph(num_nodes: int = 6) -> nx.Graph:
    graph = nx.path_graph(num_nodes)
    for node in graph.nodes():
        graph.nodes[node]["EPI"] = float(node)
        graph.nodes[node]["nu_f"] = 1.0
        graph.nodes[node]["phase"] = 0.0
    return graph


def test_fft_cache_coordinator_reuses_spectral_basis() -> None:
    coordinator = FFTCacheCoordinator()
    graph = _make_graph()

    basis_first = coordinator.get_spectral_basis(graph)
    basis_second = coordinator.get_spectral_basis(graph)

    np.testing.assert_allclose(basis_first.eigenvalues, basis_second.eigenvalues)
    stats = coordinator.get_stats()
    assert stats["spectral_hits"] >= 1
    assert stats["spectral_misses"] == 1


def test_fft_cache_coordinator_registers_results() -> None:
    coordinator = FFTCacheCoordinator()
    graph = _make_graph()

    fake_result = FFTArithmeticResult(
        operation=SpectralOperation.CONVOLUTION,
        input_shape=(graph.number_of_nodes(),),
        output_data=np.ones(graph.number_of_nodes()),
    )

    coordinator.register_fft_result(graph, fake_result, metadata={"mode": "unit-test"})
    stats = coordinator.get_stats()
    assert stats["registered_results"] == 1
