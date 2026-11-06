"""Optimized consumers must preserve raw scalar evidence before coercion."""

from copy import deepcopy
from fractions import Fraction

import networkx as nx
import pytest

from tnfr.dynamics.fft_engine import FFTDynamicsEngine
from tnfr.dynamics.nodal_optimizer import NodalEquationOptimizer
from tnfr.errors import TNFRValueError


def _graph():
    graph = nx.path_graph(2)
    for node in graph:
        graph.nodes[node].update(EPI=0.25 * node, theta=0.1 * node, nu_f=1.0)
    return graph


@pytest.mark.parametrize("invalid", [True, "0.25", Fraction(1, 2**1075)])
def test_fft_initialization_rejects_raw_phase_before_alias_conversion(invalid):
    graph = _graph()
    graph.nodes[0]["theta"] = invalid
    before = deepcopy(dict(graph.nodes(data=True)))

    with pytest.raises(TNFRValueError, match="finite real scalar"):
        FFTDynamicsEngine(enable_caching=False).create_fft_state(graph)

    assert dict(graph.nodes(data=True)) == before


@pytest.mark.parametrize("invalid", [True, "0.25", Fraction(-1, 2**1075)])
@pytest.mark.parametrize("consumer", ["step", "reconstruct", "coherence"])
def test_fft_consumers_reject_raw_capacity_without_node_writes(invalid, consumer):
    graph = _graph()
    engine = FFTDynamicsEngine(enable_caching=False)
    state = engine.create_fft_state(graph)
    graph.nodes[0]["nu_f"] = invalid
    before = deepcopy(dict(graph.nodes(data=True)))

    with pytest.raises(TNFRValueError, match="finite real scalar"):
        if consumer == "step":
            engine.fft_accelerated_step(graph, state, 0.125)
        elif consumer == "reconstruct":
            engine.reconstruct_graph_from_fft(graph, state)
        else:
            engine._compute_spectral_coherence(graph, state)

    assert dict(graph.nodes(data=True)) == before


@pytest.mark.parametrize("attribute", ["nu_f", "theta"])
@pytest.mark.parametrize("sign", [-1, 1])
def test_nodal_proposal_rejects_nonzero_scalar_lost_to_zero(attribute, sign):
    graph = _graph()
    graph.nodes[0][attribute] = Fraction(sign, 2**1075)
    before = deepcopy(dict(graph.nodes(data=True)))

    with pytest.raises(TNFRValueError, match="underflows"):
        NodalEquationOptimizer(enable_cache=False).compute_vectorized_nodal_evolution(
            graph, 0.125
        )

    assert dict(graph.nodes(data=True)) == before
