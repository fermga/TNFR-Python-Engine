"""Regression tests for scalar EPI and evidence-scoped diagnostics."""

from __future__ import annotations

import json
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from tnfr.dynamics import emergent_centralization as centralization_module
from tnfr.dynamics import self_optimizing_engine as self_optimization_module
from tnfr.dynamics.emergent_centralization import TNFREmergentCentralizationEngine
from tnfr.dynamics.self_optimizing_engine import (
    OptimizationExperience,
    SelfOptimizationResult,
    TNFRSelfOptimizingEngine,
)
from tnfr.engines.manifest import finite_json_state
from tnfr.engines.pattern_discovery import mathematical_patterns as pattern_module
from tnfr.engines.pattern_discovery.mathematical_patterns import (
    TNFREmergentPatternEngine,
)
from tnfr.errors import TNFRValueError
from tnfr.mathematics import BEPIElement
from tnfr.types import require_finite_real_scalar_epi


def _scalar_graph(values: list[float], *, alias: str = "psi") -> nx.Graph:
    graph = nx.path_graph(len(values))
    for node, value in enumerate(values):
        graph.nodes[node].update(
            {alias: value, "nu_f": 1.0, "delta_nfr": 0.0, "theta": 0.0}
        )
    return graph


def _rich_epi() -> BEPIElement:
    return BEPIElement((-2.0, -2.0), (-2.0, -1.0), (0.0, 1.0))


def test_shared_manifest_value_preserves_semantics_and_rejects_loss() -> None:
    value = {
        "measured": np.float64(1.25),
        "evaluated": np.bool_(False),
        "nested": (1, {"missing": None}),
    }

    assert finite_json_state(value) == {
        "measured": 1.25,
        "evaluated": False,
        "nested": [1, {"missing": None}],
    }
    with pytest.raises(ValueError, match="state.bad requires finite JSON state"):
        finite_json_state({"bad": np.inf})
    with pytest.raises(ValueError, match="state.bad.*unsupported object"):
        finite_json_state({"bad": object()})


def test_shared_scalar_epi_boundary_preserves_sign_and_rejects_lossy_inputs() -> None:
    signed = BEPIElement((-2.0, -2.0), (-2.0, -2.0), (0.0, 1.0))

    assert require_finite_real_scalar_epi(signed) == pytest.approx(-2.0)
    for invalid in (_rich_epi(), 1.0j, True, np.bool_(False), np.inf, "1.0"):
        with pytest.raises(TNFRValueError, match="finite uniform-real EPI"):
            require_finite_real_scalar_epi(invalid)


def test_centralization_reads_aliases_and_signed_epi_consistently() -> None:
    graph = _scalar_graph([-10.0, 0.0])
    engine = TNFREmergentCentralizationEngine()
    engine.coordination_threshold = -1.0

    candidates = engine.analyze_information_flow_centralization(graph)

    assert [candidate.node_id for candidate in candidates] == [0]
    signature = candidates[0].mathematical_signature
    assert signature["signed_epi"] == pytest.approx(-10.0)
    assert signature["epi_magnitude"] == pytest.approx(10.0)
    assert signature["information_gradient"] == pytest.approx(10.0)


def test_centralization_normalizes_extreme_same_sign_epi_without_overflow() -> None:
    maximum = np.finfo(float).max
    graph = _scalar_graph([maximum, maximum])
    engine = TNFREmergentCentralizationEngine()
    engine.coordination_threshold = -1.0

    candidates = engine.analyze_information_flow_centralization(graph)

    assert len(candidates) == 2
    assert all(
        candidate.mathematical_signature["epi_concentration"]
        == pytest.approx(0.5)
        for candidate in candidates
    )


def test_centralization_reports_scores_without_inventing_improvements() -> None:
    graph = _scalar_graph([10.0, 0.0])
    engine = TNFREmergentCentralizationEngine(enable_adaptive_topology=True)
    engine.coordination_threshold = -1.0

    result = engine.optimize_centralization(graph)
    statistics = engine.get_centralization_statistics()

    assert result.performance_improvements == {}
    assert all(
        node.current_load is None
        for pattern in result.discovered_patterns
        for node in pattern.coordination_nodes
    )
    assert result.coordination_efficiency is None
    assert result.fault_tolerance is None
    assert result.diagnostic_scores["coordination_coverage_score"] > 0.0
    assert statistics["adaptive_topology_requested"] is True
    assert statistics["adaptive_topology_applied"] is False
    assert statistics["successful_centralizations"] == 0


@pytest.mark.parametrize(
    "factory",
    [
        lambda graph: TNFREmergentCentralizationEngine().analyze_information_flow_centralization(graph),
        lambda graph: TNFREmergentPatternEngine().discover_entropy_flow_patterns(graph),
        lambda graph: _self_optimization_landscape(graph),
    ],
)
def test_diagnostics_reject_nonuniform_primary_epi_without_alias_fallback(factory) -> None:
    graph = _scalar_graph([1.0, 0.0])
    graph.nodes[0]["EPI"] = _rich_epi()
    graph.nodes[0]["psi"] = 5.0

    with pytest.raises(TNFRValueError, match="finite uniform-real EPI"):
        factory(graph)


def _self_optimization_landscape(graph: nx.Graph) -> dict[str, object]:
    engine = TNFRSelfOptimizingEngine()
    engine.pattern_engine = None
    return engine.analyze_mathematical_optimization_landscape(graph)


def test_extreme_finite_epi_reductions_do_not_overflow_or_fabricate_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    maximum = np.finfo(float).max
    graph = _scalar_graph([maximum, -maximum])
    pattern_engine = TNFREmergentPatternEngine()
    monkeypatch.setattr(pattern_module, "PATTERNS_ENTROPY_THRESHOLD_CANONICAL", -1.0)
    monkeypatch.setattr(pattern_module, "PATTERNS_DIVERGENCE_THRESHOLD_CANONICAL", -1.0)

    entropy_patterns = pattern_engine.discover_entropy_flow_patterns(graph)
    landscape = _self_optimization_landscape(graph)

    assert entropy_patterns
    signature = entropy_patterns[0].mathematical_signature
    assert signature["mean_edge_epi_contrast"] is None
    assert signature["squared_epi_norm"] is None
    assert landscape["nodal_equation_analysis"]["epi_variance"] is None


def test_self_optimization_uses_signed_alias_epi_for_variance() -> None:
    graph = _scalar_graph([-2.0, 1.0])

    analysis = _self_optimization_landscape(graph)

    assert analysis["nodal_equation_analysis"]["epi_variance"] == pytest.approx(2.25)


def test_pattern_snapshot_metrics_do_not_claim_prediction_or_compression(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    graph = _scalar_graph([9.0, 1.0, 1.0])
    monkeypatch.setattr(pattern_module, "PATTERNS_ENTROPY_THRESHOLD_CANONICAL", 0.0)
    monkeypatch.setattr(pattern_module, "PATTERNS_DIVERGENCE_THRESHOLD_CANONICAL", 0.0)
    engine = TNFREmergentPatternEngine()

    entropy_patterns = engine.discover_entropy_flow_patterns(graph)
    result = engine.discover_all_patterns(graph)

    assert entropy_patterns
    assert entropy_patterns[0].prediction_horizon is None
    assert entropy_patterns[0].compression_ratio is None
    assert entropy_patterns[0].mathematical_signature["temporal_evidence"] is False
    assert result.compression_potential is None
    assert result.predictive_accuracy is None
    assert result.emergent_optimization_strategies == []

    exported = engine.export_pattern_manifest(
        graph, result, tmp_path, partition_id="no-performance-evidence"
    )
    manifest = json.loads(exported["manifest_absolute"].read_text(encoding="utf-8"))
    summary = json.loads(exported["summary_absolute"].read_text(encoding="utf-8"))
    assert manifest["compression_potential"] is None
    assert manifest["predictive_accuracy"] is None
    assert summary["compression_potential"] is None
    assert summary["predictive_accuracy"] is None
    entropy_records = [
        item
        for item in manifest["discovered_patterns"]
        if item["pattern_type"] == "entropy_flow"
    ]
    assert entropy_records
    assert entropy_records[0]["prediction_horizon"] is None
    assert entropy_records[0]["compression_ratio"] is None
    signature = entropy_records[0]["mathematical_signature"]
    assert signature["temporal_evidence"] is False
    assert isinstance(signature["integration_hint"], dict)


def test_pattern_spectral_and_field_readers_do_not_reuse_stale_local_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _scalar_graph([1.0, 2.0])
    spectral_calls = 0

    def fake_spectrum(current: nx.Graph):
        nonlocal spectral_calls
        spectral_calls += 1
        size = current.number_of_nodes()
        return np.arange(size, dtype=float), np.eye(size)

    monkeypatch.setattr(pattern_module, "HAS_SPECTRAL", True)
    monkeypatch.setattr(pattern_module, "get_laplacian_spectrum", fake_spectrum)
    pattern_module._get_spectral_basis(graph)
    graph.add_node(2, psi=3.0)
    eigenvalues, _ = pattern_module._get_spectral_basis(graph)

    assert spectral_calls == 2
    assert eigenvalues.shape == (3,)


def _recommendation() -> SelfOptimizationResult:
    return SelfOptimizationResult(
        learned_policies=[],
        optimization_improvements={},
        recommended_strategies=["vectorized"],
        mathematical_insights={},
        predicted_speedups={},
        adaptive_configurations={},
        execution_time=0.0,
    )


class _EvidenceOrchestrator:
    def __init__(self, performance: dict[str, float | None], cache_hits: int | None):
        self.performance = performance
        self.cache_hits = cache_hits

    def analyze_optimization_profile(self, graph, operation):
        return SimpleNamespace(edge_density=nx.density(graph))

    def execute_optimization(self, graph, operation, strategy, **kwargs):
        return SimpleNamespace(
            execution_time=0.01,
            speedup_factor=1.0,
            memory_used_mb=0.0,
            cache_hits=0,
            accuracy_preserved=True,
            details={
                "performance_measurements": self.performance,
                "cache_measurements": {"hits": self.cache_hits},
            },
        )


def test_self_optimizer_does_not_learn_legacy_performance_sentinels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _scalar_graph([1.0, 2.0])
    engine = TNFRSelfOptimizingEngine()
    monkeypatch.setattr(self_optimization_module, "HAS_CONSERVATION", False)
    monkeypatch.setattr(engine, "recommend_optimization_strategy", lambda *args: _recommendation())
    engine.orchestrator = _EvidenceOrchestrator(
        {"speedup_factor": None, "memory_used_mb": None}, None
    )

    engine.optimize_automatically(graph, "diagnostic")

    assert engine.experience_history[-1].performance_metrics == {
        "execution_time": pytest.approx(0.01)
    }


def test_self_optimizer_records_authoritative_performance_measurements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = _scalar_graph([1.0, 2.0])
    engine = TNFRSelfOptimizingEngine()
    monkeypatch.setattr(self_optimization_module, "HAS_CONSERVATION", False)
    monkeypatch.setattr(engine, "recommend_optimization_strategy", lambda *args: _recommendation())
    engine.orchestrator = _EvidenceOrchestrator(
        {"speedup_factor": 2.0, "memory_used_mb": 3.5}, 4
    )

    engine.optimize_automatically(graph, "diagnostic")

    assert engine.experience_history[-1].performance_metrics == {
        "execution_time": pytest.approx(0.01),
        "speedup_factor": pytest.approx(2.0),
        "memory_used_mb": pytest.approx(3.5),
        "cache_hits": 4,
    }


def test_none_performance_evidence_is_ignored_by_policy_learning() -> None:
    engine = TNFRSelfOptimizingEngine()
    initial_config = dict(engine.adaptive_config)
    for index in range(12):
        engine.learn_from_experience(
            OptimizationExperience(
                graph_properties={"nodes": 10, "edges": 9, "density": 0.2},
                operation_type="diagnostic",
                strategy_used="nodal_vectorized",
                parameters={"backend": "numpy"},
                performance_metrics={
                    "speedup_factor": None,
                    "memory_used_mb": None,
                    "structural_charge_drift": None,
                },
                timestamp=float(index),
                success=True,
            )
        )

    assert engine.learned_policies == []
    assert engine.adaptive_config == initial_config
