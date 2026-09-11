"""Pressure-model and envelope contracts for the unified nodal backend."""

from __future__ import annotations

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.dynamics.nodal_optimizer import NodalEquationOptimizer
from tnfr.dynamics.unified_backend import (
    ComputationType,
    TNFRUnifiedBackend,
    UnifiedComputationRequest,
)


def _uniform_graph() -> nx.Graph:
    graph = nx.path_graph(2)
    for node in graph:
        graph.nodes[node][ALIAS_EPI[0]] = 0.5
        graph.nodes[node][ALIAS_VF[0]] = 1.0
        graph.nodes[node][ALIAS_DNFR[0]] = 0.4
        graph.nodes[node][ALIAS_THETA[0]] = 0.0
    return graph


def _request(graph: nx.Graph, **parameters: object) -> UnifiedComputationRequest:
    return UnifiedComputationRequest(
        computation_type=ComputationType.NODAL_EVOLUTION,
        graph=graph,
        parameters={"dt": 0.1, **parameters},
        enable_cache=False,
        optimization_level=2,
    )


def test_default_nodal_backend_never_substitutes_epi_diffusion_for_live_pressure() -> None:
    graph = _uniform_graph()
    backend = TNFRUnifiedBackend()

    result = backend._execute_nodal_evolution(_request(graph), "numpy")

    assert result["pressure_model"] == "stored_delta_nfr"
    assert result["detached"] is True
    assert result["nodal_states"][0][0] == pytest.approx(0.54)
    assert result["nodal_states"][1][0] == pytest.approx(0.54)


def test_epi_diffusion_requires_explicit_model_and_keeps_uniform_field_fixed() -> None:
    graph = _uniform_graph()
    backend = TNFRUnifiedBackend()

    result = backend._execute_nodal_evolution(
        _request(graph, pressure_model="epi_diffusion"), "numpy"
    )

    assert result["pressure_model"] == "epi_diffusion"
    assert result["nodal_states"][0][0] == pytest.approx(0.5)
    assert result["nodal_states"][1][0] == pytest.approx(0.5)


def test_temporal_integration_receives_one_normalized_nodal_envelope() -> None:
    graph = _uniform_graph()
    backend = TNFRUnifiedBackend()
    request = UnifiedComputationRequest(
        computation_type=ComputationType.TEMPORAL_INTEGRATION,
        graph=graph,
        parameters={"num_steps": 2, "dt": 0.1},
        enable_cache=False,
        return_trajectory=True,
        optimization_level=1,
    )

    result = backend._execute_temporal_integration(request, "numpy")

    assert len(result["trajectory"]) == 2
    assert result["trajectory"][0]["nodal_states"][0][0] == pytest.approx(0.54)
    assert graph.nodes[0][ALIAS_EPI[0]] == pytest.approx(0.58)


def test_optimizer_resolves_u2_roles_from_canonical_grammar_sets() -> None:
    result = NodalEquationOptimizer(enable_cache=False).optimize_operator_sequence(
        _uniform_graph(), ["VAL", "THOL", "ZHIR", "IL", "OZ", "SHA"]
    )

    assert "preserve_u2_stabilizer_order" in result["grammar_dependencies"]
    assert "stabilizer_destabilizer_fusion" not in result["optimizations"]
    assert result["caching_opportunities"] == 1

def test_large_stored_pressure_trajectory_never_routes_to_epi_diffusion() -> None:
    graph = nx.path_graph(21)
    for node in graph:
        graph.nodes[node][ALIAS_EPI[0]] = 0.5
        graph.nodes[node][ALIAS_VF[0]] = 1.0
        graph.nodes[node][ALIAS_DNFR[0]] = 0.4
        graph.nodes[node][ALIAS_THETA[0]] = 0.0
    backend = TNFRUnifiedBackend()
    request = UnifiedComputationRequest(
        computation_type=ComputationType.TEMPORAL_INTEGRATION,
        graph=graph,
        parameters={"num_steps": 1, "dt": 0.1},
        enable_cache=False,
        optimization_level=2,
    )

    result = backend._execute_temporal_integration(request, "numpy")

    assert result["pressure_model"] == "stored_delta_nfr"
    assert graph.nodes[0][ALIAS_EPI[0]] == pytest.approx(0.54)

def test_execute_computation_propagates_invalid_state_without_fallback() -> None:
    graph = _uniform_graph()
    backend = TNFRUnifiedBackend()
    request = _request(graph, dt=float("nan"))

    with pytest.raises(ValueError, match="positive finite"):
        backend.execute_computation(request)

    assert backend.get_performance_statistics()["total_computations"] == 0


def test_spectral_cache_key_tracks_live_weighted_topology() -> None:
    graph = nx.complete_graph(4)
    for left, right in graph.edges:
        graph.edges[left, right]["weight"] = 1.0
    for node in graph:
        graph.nodes[node][ALIAS_EPI[0]] = float(node)
    backend = TNFRUnifiedBackend()
    request = UnifiedComputationRequest(
        computation_type=ComputationType.SPECTRAL_ANALYSIS,
        graph=graph,
    )

    first = backend.execute_computation(request).results
    graph.edges[0, 1]["weight"] = 7.0
    second = backend.execute_computation(request).results

    assert len(backend._spectral_cache) == 2
    assert not all(
        first["eigenvectors"].flat == second["eigenvectors"].flat
    )


def test_field_cache_tracks_pressure_and_returns_defensive_results() -> None:
    graph = _uniform_graph()
    backend = TNFRUnifiedBackend()
    request = UnifiedComputationRequest(
        computation_type=ComputationType.FIELD_COMPUTATION,
        graph=graph,
    )

    first = backend.execute_computation(request).results
    expected = dict(first["phi_s"])
    first["phi_s"][0] = 999.0
    cached = backend.execute_computation(request).results
    assert cached["phi_s"] == expected

    graph.nodes[1][ALIAS_DNFR[0]] = 2.0
    changed = backend.execute_computation(request).results
    assert changed["phi_s"] != expected
    assert len(backend._field_cache) == 2


def test_operator_application_executes_one_atomic_validated_word() -> None:
    graph = _uniform_graph()
    backend = TNFRUnifiedBackend()
    request = UnifiedComputationRequest(
        computation_type=ComputationType.OPERATOR_APPLICATION,
        graph=graph,
        parameters={"node": 0, "operators": ["AL", "IL", "SHA"]},
        enable_cache=True,
    )

    result = backend.execute_computation(request).results

    assert result["applied_operators"] == [
        "emission",
        "coherence",
        "silence",
    ]
    assert result["mutated"] is True
    assert result["atomic"] is True
    assert list(graph.nodes[0]["glyph_history"])[-3:] == ["AL", "IL", "SHA"]


def test_operator_application_rejects_invalid_word_before_writing() -> None:
    graph = _uniform_graph()
    before = dict(graph.nodes[0])
    backend = TNFRUnifiedBackend()
    request = UnifiedComputationRequest(
        computation_type=ComputationType.OPERATOR_APPLICATION,
        graph=graph,
        parameters={"node": 0, "operators": ["OZ"]},
    )

    with pytest.raises(ValueError, match="Invalid canonical sequence"):
        backend.execute_computation(request)

    assert dict(graph.nodes[0]) == before


def test_facade_reports_actual_route_and_cache_evidence() -> None:
    graph = _uniform_graph()
    backend = TNFRUnifiedBackend()
    request = UnifiedComputationRequest(
        computation_type=ComputationType.SPECTRAL_ANALYSIS,
        graph=graph,
        preferred_backend="jax",
        optimization_level=2,
    )

    first = backend.execute_computation(request)
    second = backend.execute_computation(request)

    assert first.backend_used == "tnfr-spectral-api"
    assert first.optimization_strategy == "none"
    assert first.memory_used_mb is None
    assert first.cache_hits == 0
    assert first.cache_misses == 1
    assert second.cache_hits == 1
    assert second.cache_misses == 0
    assert backend.get_performance_statistics()["mathematical_backend_dispatch"] is False


def test_field_route_returns_the_complete_structural_tetrad() -> None:
    graph = _uniform_graph()
    result = TNFRUnifiedBackend().execute_computation(
        UnifiedComputationRequest(
            computation_type=ComputationType.FIELD_COMPUTATION, graph=graph
        )
    )

    assert {
        "phi_s",
        "phase_gradient",
        "phase_curvature",
        "coherence_length",
    } <= result.results.keys()
    assert result.backend_used == "tnfr-field-readers"


def test_cross_scale_route_rejects_missing_canonical_state_map() -> None:
    graph = _uniform_graph()
    request = UnifiedComputationRequest(
        computation_type=ComputationType.CROSS_SCALE_COUPLING, graph=graph
    )

    with pytest.raises(ValueError, match="no implemented canonical state map"):
        TNFRUnifiedBackend().execute_computation(request)


@pytest.mark.parametrize("level", [True, -1, 3, 1.5, "2"])
def test_optimization_level_rejects_inert_or_ambiguous_values(level: object) -> None:
    request = _request(_uniform_graph())
    request.optimization_level = level  # type: ignore[assignment]

    with pytest.raises(ValueError, match=r"integer in \[0, 2\]"):
        TNFRUnifiedBackend().execute_computation(request)


def test_internal_caches_obey_one_shared_approximate_byte_budget() -> None:
    graph = nx.path_graph(12)
    for node in graph:
        graph.nodes[node][ALIAS_EPI[0]] = float(node)
        graph.nodes[node][ALIAS_VF[0]] = 1.0
        graph.nodes[node][ALIAS_DNFR[0]] = 0.0
        graph.nodes[node][ALIAS_THETA[0]] = 0.0
    backend = TNFRUnifiedBackend(cache_size_mb=0.0001)

    backend.execute_computation(
        UnifiedComputationRequest(
            computation_type=ComputationType.SPECTRAL_ANALYSIS, graph=graph
        )
    )
    backend.execute_computation(
        UnifiedComputationRequest(
            computation_type=ComputationType.FIELD_COMPUTATION, graph=graph
        )
    )

    assert backend._cache_size_bytes <= backend._cache_budget_bytes
    assert not backend._spectral_cache
    assert not backend._field_cache


@pytest.mark.parametrize("budget", [True, 0, -1, float("nan"), float("inf"), "bad"])
def test_cache_budget_must_be_a_positive_finite_scalar(budget: object) -> None:
    with pytest.raises((TypeError, ValueError), match="positive finite"):
        TNFRUnifiedBackend(cache_size_mb=budget)  # type: ignore[arg-type]
