"""Regression tests for the nodal-optimizer spectral path.

Guards two canonical properties of
:mod:`tnfr.dynamics.nodal_optimizer`:

1. ``precompute_spectral_basis`` executes without error. Before the
   structural-Laplacian canonicity fix it called
   ``get_laplacian_spectrum(G, normalized=True, cache_key=...)`` with keyword
   arguments the function never accepted, so the whole spectral path raised
   ``TypeError`` and was silently dead / untested.
2. The precomputed spectrum is the canonical EMERGENT operator -- the symmetric
   normalized Laplacian ``L_sym`` (which shares the spectrum of the random-walk
   diffusion operator ``L_rw = I - D^-1 W`` that the canonical ΔNFR realises) --
   not the imposed combinatorial graph Laplacian ``D - A``.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import networkx as nx
import numpy as np
import pytest

from tnfr.errors import TNFRValueError
from tnfr.mathematics import BEPIElement
from tnfr.physics.structural_diffusion import structural_diffusion_operator

from tnfr.dynamics.nodal_optimizer import (
    HAS_SPECTRAL,
    NodalEquationOptimizer,
    NodalOptimizationState,
)

pytestmark = pytest.mark.skipif(
    not HAS_SPECTRAL, reason="spectral dependencies (scipy) required"
)


def _graph() -> nx.Graph:
    G = nx.karate_club_graph()
    for node in G.nodes():
        G.nodes[node]["EPI"] = float(node % 5) / 5.0
        G.nodes[node]["vf"] = 1.0
        G.nodes[node]["theta"] = 0.0
    return G


def test_precompute_spectral_basis_executes() -> None:
    """The spectral path runs (regression: previously raised TypeError)."""
    opt = NodalEquationOptimizer(enable_cache=False)
    state = opt.precompute_spectral_basis(_graph())

    assert isinstance(state, NodalOptimizationState)
    n = 34  # karate club
    assert state.eigenvalues.shape == (n,)
    assert state.eigenvectors.shape == (n, n)
    assert state.node_index and len(state.node_index) == n


def test_precomputed_spectrum_is_canonical_emergent_operator() -> None:
    """The spectrum equals L_sym (emergent), not combinatorial D - A."""
    from tnfr.physics.structural_diffusion import symmetric_normalized_laplacian

    G = _graph()
    opt = NodalEquationOptimizer(enable_cache=False)
    state = opt.precompute_spectral_basis(G)

    _, l_sym = symmetric_normalized_laplacian(G)
    sym_spectrum = np.sort(np.linalg.eigvalsh(l_sym))
    comb_spectrum = np.sort(
        np.linalg.eigvalsh(nx.laplacian_matrix(G).toarray().astype(float))
    )

    got = np.sort(np.real(state.eigenvalues))
    # Matches the canonical emergent L_sym spectrum ...
    assert np.allclose(got, sym_spectrum, atol=1e-8)
    # ... and is genuinely a different operator from combinatorial D - A.
    assert not np.allclose(got, comb_spectrum, atol=1e-6)


def test_spectral_basis_cache_returns_detached_read_only_snapshots() -> None:
    """A cache hit reuses work without exposing the owned state."""
    G = _graph()
    opt = NodalEquationOptimizer(enable_cache=True)
    first = opt.precompute_spectral_basis(G)
    second = opt.precompute_spectral_basis(G)

    assert first is not second
    np.testing.assert_array_equal(first.eigenvalues, second.eigenvalues)
    assert opt.get_optimization_stats()["cache_misses"] == 1
    assert opt.get_optimization_stats()["cache_hits"] == 1
    with pytest.raises(ValueError):
        first.eigenvalues[0] = 99.0
    with pytest.raises(TypeError):
        first.node_index[next(iter(G))] = 99


def _irregular_weighted_graph(graph_type=nx.Graph) -> nx.Graph:
    graph = graph_type()
    graph.add_nodes_from(("hub", 7, ("leaf", 1), "isolated"))
    graph.add_edge("hub", 7, weight=1.0)
    graph.add_edge("hub", ("leaf", 1), weight=3.0)
    values = {
        "hub": (1.0, 0.5),
        7: (-0.25, 2.0),
        ("leaf", 1): (0.75, 1.25),
        "isolated": (4.0, 9.0),
    }
    for node, (epi, frequency) in values.items():
        graph.nodes[node].update(EPI=epi, nu_f=frequency, theta=0.0)
    return graph


def test_nodal_proposal_matches_pointwise_heterogeneous_lrw_oracle() -> None:
    graph = _irregular_weighted_graph()
    optimizer = NodalEquationOptimizer(enable_cache=True)
    dt = 0.05

    proposal = optimizer.compute_vectorized_nodal_evolution(graph, dt)

    nodes, laplacian = structural_diffusion_operator(graph)
    epi = np.array([graph.nodes[node]["EPI"] for node in nodes])
    frequency = np.array([graph.nodes[node]["nu_f"] for node in nodes])
    expected = epi + dt * frequency * (-(laplacian @ epi))
    np.testing.assert_allclose(
        [proposal[node][0] for node in nodes],
        expected,
        rtol=0.0,
        atol=1e-14,
    )
    assert proposal["isolated"][0] == pytest.approx(4.0)


def test_nodal_cache_reads_live_frequency_and_invalidates_on_weight_change() -> None:
    graph = _irregular_weighted_graph()
    optimizer = NodalEquationOptimizer(enable_cache=True)

    first_state = optimizer.precompute_spectral_basis(graph)
    first = optimizer.compute_vectorized_nodal_evolution(graph, 0.05)
    graph.nodes[7]["nu_f"] = 9.0
    second = optimizer.compute_vectorized_nodal_evolution(graph, 0.05)
    assert second[7][0] != pytest.approx(first[7][0])
    live_state = optimizer.precompute_spectral_basis(graph)
    assert live_state is not first_state
    assert live_state.vf_vector[live_state.node_index[7]] == pytest.approx(9.0)

    graph.edges["hub", 7]["weight"] = 5.0
    second_state = optimizer.precompute_spectral_basis(graph)
    assert second_state is not first_state
    assert not np.array_equal(
        second_state.diffusion_operator,
        first_state.diffusion_operator,
    )


def test_nodal_directed_fallback_uses_actual_random_walk_operator() -> None:
    graph = _irregular_weighted_graph(nx.DiGraph)
    optimizer = NodalEquationOptimizer(enable_cache=True)
    proposal = optimizer.compute_vectorized_nodal_evolution(graph, 0.1)

    nodes, laplacian = structural_diffusion_operator(graph)
    epi = np.array([graph.nodes[node]["EPI"] for node in nodes])
    frequency = np.array([graph.nodes[node]["nu_f"] for node in nodes])
    expected = epi + 0.1 * frequency * (-(laplacian @ epi))
    np.testing.assert_allclose(
        [proposal[node][0] for node in nodes],
        expected,
        rtol=0.0,
        atol=1e-14,
    )
    assert optimizer.precompute_spectral_basis(graph).eigenvalues.size == 0


def test_nodal_phase_prediction_excludes_u3_incompatible_neighbor() -> None:
    graph = nx.path_graph(2)
    graph.graph["DELTA_PHI_MAX"] = 0.1
    graph.nodes[0].update(EPI=0.0, nu_f=0.0, theta=0.0)
    graph.nodes[1].update(EPI=0.0, nu_f=0.0, theta=2.0)

    proposal = NodalEquationOptimizer(enable_cache=False).compute_vectorized_nodal_evolution(
        graph, 0.1
    )

    assert proposal[0][1] == pytest.approx(0.0)
    assert proposal[1][1] == pytest.approx(2.0)


def test_nodal_optimizer_rejects_nonuniform_bepi_and_invalid_dt() -> None:
    graph = _irregular_weighted_graph()
    graph.nodes["hub"]["EPI"] = BEPIElement(
        (-0.8, -0.7),
        (-0.8, -0.8),
        (0.0, 1.0),
    )
    optimizer = NodalEquationOptimizer(enable_cache=False)

    with pytest.raises(TNFRValueError, match="scalar EPI"):
        optimizer.compute_vectorized_nodal_evolution(graph, 0.1)
    with pytest.raises(TNFRValueError, match="positive"):
        optimizer.compute_vectorized_nodal_evolution(_graph(), 0.0)


def test_clear_optimization_cache_preserves_graph_id_compatibility() -> None:
    graph = _graph()
    optimizer = NodalEquationOptimizer(enable_cache=True)
    optimizer.precompute_spectral_basis(graph)
    assert optimizer.get_optimization_stats()["cached_graphs"] == 1

    optimizer.clear_optimization_cache(id(graph))

    assert optimizer.get_optimization_stats()["cached_graphs"] == 0

def test_nodal_optimizer_public_state_cannot_poison_internal_cache() -> None:
    graph = _irregular_weighted_graph()
    optimizer = NodalEquationOptimizer(enable_cache=True)
    state = optimizer.precompute_spectral_basis(graph)

    with pytest.raises(FrozenInstanceError):
        state.diffusion_operator = np.zeros_like(state.diffusion_operator)
    with pytest.raises(ValueError):
        state.diffusion_operator[0, 0] = 99.0

    proposal = optimizer.compute_vectorized_nodal_evolution(graph, 0.037)
    assert set(proposal) == set(graph)


def test_nodal_phase_gate_change_is_never_hidden_by_result_cache() -> None:
    graph = nx.path_graph(2)
    for node, phase in enumerate((0.0, 0.5)):
        graph.nodes[node].update(EPI=0.0, nu_f=0.0, theta=phase)
    optimizer = NodalEquationOptimizer(enable_cache=False)

    graph.graph["DELTA_PHI_MAX"] = 1.0
    coupled = optimizer.compute_vectorized_nodal_evolution(graph, 0.077)
    graph.graph["DELTA_PHI_MAX"] = 0.1
    blocked = optimizer.compute_vectorized_nodal_evolution(graph, 0.077)

    assert coupled[0][1] != pytest.approx(0.0)
    assert coupled[1][1] != pytest.approx(0.5)
    assert blocked[0][1] == pytest.approx(0.0)
    assert blocked[1][1] == pytest.approx(0.5)

def test_live_nodal_step_does_not_precompute_unused_spectrum() -> None:
    graph = _irregular_weighted_graph()
    optimizer = NodalEquationOptimizer(enable_cache=True)

    optimizer.compute_vectorized_nodal_evolution(graph, 0.01)
    state = optimizer._optimization_states[graph]

    assert state.eigenvalues.size == 0
    assert state.eigenvectors.shape == (len(graph), 0)


def test_operator_plan_reports_candidates_without_fabricated_speedup() -> None:
    graph = _irregular_weighted_graph()
    optimizer = NodalEquationOptimizer(enable_cache=False)

    plan = optimizer.optimize_operator_sequence(
        graph,
        ["AL", "IL", "IL", "IL", "UM", "RA", "SHA"],
    )

    assert plan["predicted_speedup"] is None
    assert plan["speedup_evidence"] == "not_benchmarked"
    assert plan["spectral_opportunities"] == 0
    assert set(plan["candidates"]) == {
        "batch_coherence_readout_candidate",
        "shared_u3_phase_stage_candidate",
    }


def test_nodal_optimizer_enforces_graph_cache_limit() -> None:
    optimizer = NodalEquationOptimizer(enable_cache=True, max_cache_size=2)
    graphs = []
    for weight in (1.0, 2.0, 3.0):
        graph = nx.path_graph(2)
        graph.edges[0, 1]["weight"] = weight
        for node in graph:
            graph.nodes[node].update(EPI=float(node), nu_f=1.0, theta=0.0)
        graphs.append(graph)
        optimizer.precompute_spectral_basis(graph)

    stats = optimizer.get_optimization_stats()
    assert stats["cached_graphs"] == 2
    assert stats["max_cache_size"] == 2
    assert stats["cache_evictions"] == 1


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_nodal_optimizer_rejects_invalid_cache_limit(limit: object) -> None:
    with pytest.raises(TNFRValueError, match="positive integer"):
        NodalEquationOptimizer(max_cache_size=limit)  # type: ignore[arg-type]


def test_nodal_optimizer_rejects_overflowed_epi_proposal() -> None:
    graph = nx.path_graph(2)
    graph.nodes[0].update(EPI=1.0, nu_f=1e308, theta=0.0)
    graph.nodes[1].update(EPI=0.0, nu_f=0.0, theta=0.0)

    with pytest.raises(TNFRValueError, match="EPI proposal must remain finite"):
        NodalEquationOptimizer(enable_cache=False).compute_vectorized_nodal_evolution(
            graph, 1e308
        )


def test_nodal_optimizer_rejects_overflowed_phase_proposal() -> None:
    graph = nx.empty_graph(1)
    graph.nodes[0].update(EPI=0.0, nu_f=1e308, theta=0.0)

    with pytest.raises(TNFRValueError, match="phase proposal must remain finite"):
        NodalEquationOptimizer(enable_cache=False).compute_vectorized_nodal_evolution(
            graph, 1e308
        )


def test_arbitrary_euler_proposal_exposes_missing_stability_certificate() -> None:
    proposal = NodalEquationOptimizer(
        enable_cache=False
    ).compute_vectorized_nodal_evolution(_irregular_weighted_graph(), 0.01)

    assert proposal.stability_not_certified is True
    assert proposal.metadata["stability_not_certified"] is True
    assert proposal.metadata["integration_method"] == "explicit_euler"
    with pytest.raises(TypeError):
        proposal.metadata["stability_not_certified"] = False
