"""Local and global coherence share scalar admission, not neighborhood scope."""

from copy import deepcopy
from fractions import Fraction
from sys import float_info

import networkx as nx
import numpy as np
import pytest

from tnfr.constants.aliases import ALIAS_DEPI, ALIAS_DNFR
from tnfr.metrics.common import compute_coherence
from tnfr.metrics.local_coherence import (
    compute_local_coherence_fallback,
    compute_radius_structural_coherence,
)


@pytest.mark.parametrize("aliases", [ALIAS_DNFR, ALIAS_DEPI])
@pytest.mark.parametrize(
    "invalid", [None, True, np.bool_(False), "1.0", Fraction(1, 2**1075), np.nan]
)
def test_local_observations_reject_raw_invalid_authoritative_alias(aliases, invalid):
    graph = nx.path_graph(2)
    graph.nodes[1][aliases[0]] = invalid
    graph.nodes[1][aliases[1]] = 0.0
    for observe in (
        compute_coherence,
        lambda g: compute_radius_structural_coherence(g, 0),
        lambda g: compute_local_coherence_fallback(g, 0),
    ):
        with pytest.raises((TypeError, ValueError)):
            observe(graph)


def test_selected_scope_and_missing_alias_conventions_remain_distinct():
    graph = nx.path_graph(3)
    graph.nodes[0].update({ALIAS_DNFR[0]: 6.0, ALIAS_DEPI[0]: 0.0})
    graph.nodes[1].update({ALIAS_DNFR[1]: -2.0, ALIAS_DEPI[1]: 2.0})
    original = deepcopy((graph.graph, dict(graph.nodes(data=True)), list(graph.edges)))

    assert compute_radius_structural_coherence(graph, 0, 0) == pytest.approx(1 / 7)
    assert compute_radius_structural_coherence(graph, 0, 1) == pytest.approx(1 / 6)
    assert compute_local_coherence_fallback(graph, 0) == pytest.approx(1 / 5)
    assert compute_coherence(graph) == pytest.approx(3 / 13)
    assert (graph.graph, dict(graph.nodes(data=True)), list(graph.edges)) == original

    graph.nodes[2][ALIAS_DNFR[0]] = "invalid outside the selected ball"
    assert compute_radius_structural_coherence(graph, 0, 1) == pytest.approx(1 / 6)
    assert compute_local_coherence_fallback(graph, 0) == pytest.approx(1 / 5)
    with pytest.raises(TypeError):
        compute_coherence(graph)


def test_neighbor_fallback_preserves_finite_mean_at_binary64_limit():
    graph = nx.star_graph(2)
    for node in (1, 2):
        graph.nodes[node][ALIAS_DNFR[0]] = float_info.max
    expected = float(1 / (1 + Fraction.from_float(float_info.max)))
    assert expected > 0.0
    assert compute_local_coherence_fallback(graph, 0) == expected


def test_empty_and_isolated_support_preserve_their_public_scopes():
    assert compute_coherence(nx.Graph(), return_means=True) == (0.0, 0.0, 0.0)
    graph = nx.empty_graph(1)
    graph.nodes[0][ALIAS_DNFR[0]] = 2.0
    assert compute_radius_structural_coherence(graph, 0) == pytest.approx(1 / 3)
    assert compute_local_coherence_fallback(graph, 0) == 0.0
