"""Ordered signs are separate from graph mode validity and sign domains."""

import math

import networkx as nx
import numpy as np

from tnfr.physics.structural_diffusion import (
    nodal_domain_count,
    symmetric_normalized_laplacian,
    verify_discrete_modes,
)


def test_same_p4_fiedler_mode_has_order_dependent_legacy_sign_statistic():
    mode = {0: 1.0, 1: 1 / math.sqrt(2), 2: -1 / math.sqrt(2), 3: -1.0}
    counts = []
    for order in ((0, 1, 2, 3), (0, 2, 1, 3)):
        graph = nx.Graph()
        graph.add_nodes_from(order)
        graph.add_edges_from(((0, 1), (1, 2), (2, 3)))
        nodes, laplacian = symmetric_normalized_laplacian(graph)
        vector = np.array([mode[node] for node in nodes])
        np.testing.assert_allclose(laplacian @ vector, 0.5 * vector, atol=1e-15)

        # The same signed graph field always has two connected sign domains.
        domains = sum(
            nx.number_connected_components(
                graph.subgraph(node for node in graph if sign * mode[node] > 0)
            )
            for sign in (-1, 1)
        )
        assert domains == 2
        counts.append(nodal_domain_count(vector))

    assert counts == [1, 3]


def test_edgeless_graph_has_valid_modes_despite_false_ordered_sign_heuristic():
    result = verify_discrete_modes(nx.empty_graph(3))

    assert result.n_modes == 3
    assert result.spectrum_is_discrete
    assert result.modes_orthonormal
    assert result.has_uniform_zero_mode
    assert result.matches_diffusion_spectrum
    assert result.standing_wave_frequencies == (0.0, 0.0, 0.0)
    assert not result.nodal_domains_grow
    assert result.is_valid_discrete_modes
    assert "[VALID]" in result.summary()
    assert "ordered-sign heuristic=False" in result.summary()
