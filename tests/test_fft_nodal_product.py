"""Focused contracts for the FFT nodal-equation path."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from tnfr.alias import get_attr, set_attr
from tnfr.constants.aliases import (
    ALIAS_D2EPI,
    ALIAS_DEPI,
    ALIAS_DNFR,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.dynamics.fft_engine import FFTDynamicsEngine
from tnfr.errors import TNFRValueError
from tnfr.mathematics.spectral import gft, igft
from tnfr.metrics.common import compute_coherence
from tnfr.physics.structural_diffusion import (
    structural_diffusion_operator,
    structural_field,
)


def _graph(frequencies: list[float]) -> nx.Graph:
    graph = nx.path_graph(4)
    for node, (epi, frequency, phase) in enumerate(
        zip((1.0, 0.2, -0.4, 0.7), frequencies, (0.0, 0.2, -0.1, 0.4))
    ):
        graph.nodes[node]["EPI"] = epi
        set_attr(graph.nodes[node], ALIAS_VF, frequency)
        set_attr(graph.nodes[node], ALIAS_THETA, phase)
    return graph


@pytest.mark.parametrize(
    "frequencies",
    (
        [1.7, 1.7, 1.7, 1.7],
        [0.25, 0.75, 1.5, 2.25],
    ),
    ids=("homogeneous", "heterogeneous"),
)
def test_fft_step_is_exact_transform_of_the_pointwise_nodal_product(
    frequencies: list[float],
) -> None:
    graph = _graph(frequencies)
    engine = FFTDynamicsEngine(enable_caching=False)
    state = engine.create_fft_state(graph)
    dt = 0.037

    nodes, laplacian = structural_diffusion_operator(graph)
    epi_before = structural_field(graph, nodes)
    pressure = -(laplacian @ epi_before)
    expected_rate = np.asarray(frequencies) * pressure
    expected_epi = epi_before + dt * expected_rate

    updated = engine.fft_accelerated_step(graph, state, dt)
    realised_epi = igft(updated.spectral_epi, updated.eigenvectors)

    assert np.allclose(realised_epi, expected_epi, atol=2e-15, rtol=2e-15)
    realised_rate = (realised_epi - epi_before) / dt
    assert np.allclose(realised_rate, expected_rate, atol=3e-14, rtol=3e-14)
    # The state residual and this independent oracle cross two separate
    # orthonormal round trips, so compare both with the numerical contract
    # instead of requiring their last few floating-point bits to coincide.
    assert updated.nodal_residual < 2e-13
    assert float(np.max(np.abs(realised_rate - expected_rate))) < 2e-13

    # Regression counterexample: modewise multiplication is a different
    # operation and does not satisfy the physical-space nodal equation.
    pressure_hat = gft(pressure, state.eigenvectors)
    broken_rate_hat = gft(
        np.asarray(frequencies), state.eigenvectors
    ) * pressure_hat
    broken_epi = epi_before + dt * igft(broken_rate_hat, state.eigenvectors)
    assert not np.allclose(broken_epi, expected_epi, atol=1e-8, rtol=1e-8)


def test_weighted_irregular_graph_matches_canonical_lrw_and_freezes_isolate() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(range(5))
    graph.add_weighted_edges_from(((0, 1, 2.0), (1, 2, 0.5), (1, 3, 3.0)))
    epi = (1.2, -0.3, 0.8, -1.1, 4.5)
    frequencies = np.array((0.2, 0.7, 1.3, 2.1, 9.0))
    for node in graph:
        graph.nodes[node]["EPI"] = epi[node]
        set_attr(graph.nodes[node], ALIAS_VF, frequencies[node])
        set_attr(graph.nodes[node], ALIAS_THETA, 0.1 * node)

    nodes, laplacian = structural_diffusion_operator(graph)
    field = structural_field(graph, nodes)
    expected_pressure = -(laplacian @ field)
    expected_rate = frequencies * expected_pressure
    engine = FFTDynamicsEngine(enable_caching=False)
    state = engine.create_fft_state(graph)
    updated = engine.fft_accelerated_step(graph, state, 0.025)
    realised = igft(updated.spectral_epi, updated.eigenvectors)

    assert tuple(nodes) == state.node_order
    assert np.array_equal(state.diffusion_operator, laplacian)
    assert np.allclose(realised, field + 0.025 * expected_rate, atol=3e-15)
    realised_rate = (realised - field) / 0.025
    assert np.allclose(realised_rate, expected_rate, atol=2e-14)
    assert expected_pressure[4] == 0.0
    assert realised[4] == pytest.approx(field[4], abs=2e-15)

    # Directly transforming raw EPI with L_sym is not canonical on this
    # irregular weighted graph; this guards the missing similarity transform.
    symmetric_pressure = igft(
        -state.eigenvalues * state.spectral_epi, state.eigenvectors
    )
    assert not np.allclose(symmetric_pressure[:4], expected_pressure[:4])


def test_reconstruction_reports_final_prediction_but_restarts_rate_evidence() -> None:
    graph = _graph([0.4, 0.8, 1.2, 1.6])
    graph.graph["_t"] = 3.0
    graph.graph["_epi_hist"] = deque(
        ({node: -2.0 for node in graph}, {node: -1.0 for node in graph}),
        maxlen=8,
    )
    for node in graph:
        graph.nodes[node]["epi_time_history"] = deque(
            ((1.0, -1.0), (2.0, float(graph.nodes[node]["EPI"]))), maxlen=4
        )
        graph.nodes[node]["epi_history"] = [-1.0, float(graph.nodes[node]["EPI"])]
        graph.nodes[node]["_epi_history"] = deque(
            (-1.0, float(graph.nodes[node]["EPI"])), maxlen=6
        )
        set_attr(graph.nodes[node], ALIAS_D2EPI, 99.0)

    engine = FFTDynamicsEngine(enable_caching=False)
    state = engine.create_fft_state(graph)
    final_state = engine.fft_accelerated_step(graph, state, 0.125)
    nodes, laplacian = structural_diffusion_operator(graph)
    final_epi = igft(final_state.spectral_epi, final_state.eigenvectors)
    final_pressure = -(laplacian @ final_epi)
    engine.reconstruct_graph_from_fft(graph, final_state)

    assert graph.graph["_t"] == pytest.approx(3.125)
    assert len(graph.graph["_epi_hist"]) == 1
    for index, node in enumerate(final_state.node_order):
        epi = float(graph.nodes[node]["EPI"])
        assert list(graph.nodes[node]["epi_time_history"]) == [(3.125, epi)]
        assert graph.nodes[node]["epi_history"] == [epi]
        assert list(graph.nodes[node]["_epi_history"]) == [epi]
        assert get_attr(graph.nodes[node], ALIAS_DNFR, strict=True) == pytest.approx(
            final_pressure[index]
        )
        assert get_attr(graph.nodes[node], ALIAS_DEPI, strict=True) == pytest.approx(
            [0.4, 0.8, 1.2, 1.6][index] * final_pressure[index]
        )
        assert not any(key in graph.nodes[node] for key in ALIAS_D2EPI)


class _CountingCoordinator:
    def __init__(self) -> None:
        self.calls = 0
        self.hits = 0
        self._basis: SimpleNamespace | None = None

    def get_stats(self) -> dict[str, int]:
        return {"spectral_hits": self.hits}

    def get_spectral_basis(
        self, graph: nx.Graph, *, force_recompute: bool = False
    ) -> SimpleNamespace:
        from tnfr.mathematics.spectral import get_laplacian_spectrum

        self.calls += 1
        if self._basis is not None and not force_recompute:
            self.hits += 1
            return self._basis
        eigenvalues, eigenvectors = get_laplacian_spectrum(graph)
        self._basis = SimpleNamespace(
            eigenvalues=eigenvalues, eigenvectors=eigenvectors
        )
        return self._basis


def test_basis_cache_ignores_nodal_state_and_mutating_runs_are_never_result_cached() -> None:
    graph = _graph([0.5, 0.75, 1.0, 1.25])
    coordinator = _CountingCoordinator()
    engine = FFTDynamicsEngine(enable_caching=True, cache_coordinator=coordinator)

    first = engine.run_fft_simulation(graph, 1, dt=0.05)
    epi_after_first = np.array([graph.nodes[node]["EPI"] for node in graph])
    second = engine.run_fft_simulation(graph, 1, dt=0.05)
    epi_after_second = np.array([graph.nodes[node]["EPI"] for node in graph])

    assert coordinator.calls == 1
    assert engine.cache_hits == 1
    assert first["fft_operations"] == second["fft_operations"] == 1
    assert first["cache_hits"] == 0
    assert second["cache_hits"] == 1
    assert engine.total_operations == engine.fft_operations == 2
    assert graph.graph["_t"] == pytest.approx(0.1)
    assert not np.array_equal(epi_after_first, epi_after_second)
    assert not hasattr(FFTDynamicsEngine.run_fft_simulation, "_is_cached")


def test_repeated_runs_are_reproducible_in_state_trajectory_and_residual() -> None:
    first_graph = _graph([0.3, 0.7, 1.1, 1.9])
    second_graph = _graph([0.3, 0.7, 1.1, 1.9])
    first_engine = FFTDynamicsEngine(enable_caching=False)
    second_engine = FFTDynamicsEngine(enable_caching=False)

    first = first_engine.run_fft_simulation(
        first_graph, 17, dt=0.013, return_trajectory=True
    )
    second = second_engine.run_fft_simulation(
        second_graph, 17, dt=0.013, return_trajectory=True
    )

    first_epi = np.array([first_graph.nodes[node]["EPI"] for node in first_graph])
    second_epi = np.array([second_graph.nodes[node]["EPI"] for node in second_graph])
    first_phase = np.array(
        [get_attr(first_graph.nodes[node], ALIAS_THETA) for node in first_graph]
    )
    second_phase = np.array(
        [get_attr(second_graph.nodes[node], ALIAS_THETA) for node in second_graph]
    )
    assert np.array_equal(first_epi, second_epi)
    assert np.array_equal(first_phase, second_phase)
    assert first["trajectory"] == second["trajectory"]
    assert first["final_time"] == second["final_time"]
    assert first["final_coherence"] == second["final_coherence"]
    assert first["max_nodal_residual"] == second["max_nodal_residual"]
    assert first["trajectory"][-1]["time"] == pytest.approx(first["final_time"])
    assert second["trajectory"][-1]["time"] == pytest.approx(second["final_time"])
    assert sum(
        sample["time"] == first["final_time"] for sample in first["trajectory"]
    ) == 1


def test_fixed_basis_rejects_topology_changes_before_telemetry_moves() -> None:
    graph = _graph([1.0, 1.0, 1.0, 1.0])
    engine = FFTDynamicsEngine(enable_caching=False)
    state = engine.create_fft_state(graph)
    graph.add_edge(0, 3, weight=0.5)

    with pytest.raises(TNFRValueError, match="topology or edge weights"):
        engine.fft_accelerated_step(graph, state, 0.01)

    assert engine.total_operations == 0
    assert engine.fft_operations == 0


def _graph_surface(graph: nx.Graph) -> tuple[object, object, object, object]:
    return (
        tuple((node, deepcopy(dict(graph.nodes[node]))) for node in graph.nodes),
        tuple((left, right, deepcopy(dict(data))) for left, right, data in graph.edges(data=True)),
        deepcopy(dict(graph.graph)),
        getattr(graph, "_last_operator_applied", None),
    )


def test_fft_reconstruction_is_atomic_on_phase_commit_failure() -> None:
    graph = nx.path_graph(2)
    for node, (epi, phase) in enumerate(((0.1, 0.0), (0.8, 0.2))):
        graph.nodes[node].update(EPI=epi, nu_f=1.0, theta=phase)
    engine = FFTDynamicsEngine(enable_caching=False)
    state = engine.fft_accelerated_step(graph, engine.create_fft_state(graph), 0.1)
    graph.graph["_trig_version"] = "bad"
    before = _graph_surface(graph)

    with pytest.raises((TypeError, ValueError)):
        engine.reconstruct_graph_from_fft(graph, state)

    assert _graph_surface(graph) == before


def test_fft_zero_step_run_is_an_exact_graph_noop() -> None:
    graph = _graph([0.3, 0.7, 1.1, 1.9])
    graph.graph["custom"] = {"nested": [1, 2]}
    before = _graph_surface(graph)

    result = FFTDynamicsEngine(enable_caching=False).run_fft_simulation(
        graph, 0, dt=0.05, return_trajectory=True
    )

    assert result["status"] == "success"
    assert result["fft_operations"] == 0
    assert _graph_surface(graph) == before


def test_fft_step_rejects_tampered_diffusion_operator() -> None:
    graph = _graph([0.3, 0.7, 1.1, 1.9])
    engine = FFTDynamicsEngine(enable_caching=False)
    state = engine.create_fft_state(graph)
    state.diffusion_operator = np.zeros_like(state.diffusion_operator)

    with pytest.raises(TNFRValueError, match="does not match its graph basis"):
        engine.fft_accelerated_step(graph, state, 0.01)


def test_fft_phase_step_respects_live_u3_gate() -> None:
    graph = nx.path_graph(2)
    graph.graph["DELTA_PHI_MAX"] = 0.1
    graph.nodes[0].update(EPI=0.2, nu_f=0.0, theta=0.0)
    graph.nodes[1].update(EPI=0.2, nu_f=0.0, theta=0.5)
    engine = FFTDynamicsEngine(enable_caching=False)
    state = engine.fft_accelerated_step(graph, engine.create_fft_state(graph), 0.1)
    engine.reconstruct_graph_from_fft(graph, state)

    assert get_attr(graph.nodes[0], ALIAS_THETA) == pytest.approx(0.0, abs=1e-14)
    assert get_attr(graph.nodes[1], ALIAS_THETA) == pytest.approx(0.5, abs=1e-14)


def test_fft_basis_cache_invalidates_when_edge_weight_changes() -> None:
    from tnfr.dynamics.fft_cache_coordinator import FFTCacheCoordinator

    graph = nx.complete_graph(4)
    for left, right in graph.edges:
        graph.edges[left, right]["weight"] = 1.0
    coordinator = FFTCacheCoordinator()
    first = coordinator.get_spectral_basis(graph)
    graph.edges[0, 1]["weight"] = 7.0
    second = coordinator.get_spectral_basis(graph)

    assert second is not first
    assert second.signature != first.signature
    assert not np.array_equal(second.eigenvectors, first.eigenvectors)

def test_fft_state_rejects_replaced_eigenbasis() -> None:
    graph = _graph([0.4, 0.8, 1.2, 1.6])
    engine = FFTDynamicsEngine(enable_caching=False)
    state = engine.create_fft_state(graph)
    state.eigenvectors = np.eye(len(graph))

    with pytest.raises(TNFRValueError, match="immutable basis snapshot"):
        engine.fft_accelerated_step(graph, state, 0.01)


def test_cached_fft_basis_is_readonly_and_cannot_poison_later_states() -> None:
    graph = _graph([0.4, 0.8, 1.2, 1.6])
    engine = FFTDynamicsEngine(enable_caching=True)
    first = engine.create_fft_state(graph)
    baseline = np.array(first.eigenvectors, copy=True)

    with pytest.raises(ValueError):
        first.eigenvectors[0, 0] += 1.0

    second = engine.create_fft_state(graph)
    assert np.array_equal(second.eigenvectors, baseline)
    assert second.eigenbasis_digest == first.eigenbasis_digest

def test_fft_reports_canonical_coherence_separately_from_phase_sync() -> None:
    graph = _graph([4.0, 4.0, 4.0, 4.0])
    for node in graph:
        set_attr(graph.nodes[node], ALIAS_THETA, 0.0)
    engine = FFTDynamicsEngine(enable_caching=False)

    result = engine.run_fft_simulation(
        graph, 1, dt=0.01, return_trajectory=True
    )

    assert result["final_phase_sync"] == pytest.approx(1.0, abs=1e-14)
    assert result["final_coherence"] < 1.0
    assert result["final_coherence"] == pytest.approx(compute_coherence(graph))
    assert result["trajectory"][0]["phase_sync"] == pytest.approx(1.0)
    assert result["trajectory"][0]["coherence"] < 1.0

def test_fft_coordinator_basis_is_readonly_and_cache_safe() -> None:
    from tnfr.dynamics.fft_cache_coordinator import FFTCacheCoordinator

    graph = _graph([0.4, 0.8, 1.2, 1.6])
    coordinator = FFTCacheCoordinator()
    first = coordinator.get_spectral_basis(graph)
    baseline_vectors = np.array(first.eigenvectors, copy=True)
    baseline_values = np.array(first.eigenvalues, copy=True)

    with pytest.raises(ValueError):
        first.eigenvectors[0, 0] += 1.0
    with pytest.raises(ValueError):
        first.eigenvalues[0] += 1.0

    second = coordinator.get_spectral_basis(graph)
    assert np.array_equal(second.eigenvectors, baseline_vectors)
    assert np.array_equal(second.eigenvalues, baseline_values)