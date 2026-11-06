"""Tests for TNFR self-optimization dry-run support."""

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest

from tnfr.dynamics.optimization_orchestrator import (
    OptimizationResult,
    OptimizationStrategy,
)
from tnfr.dynamics.self_optimizing_engine import (
    OptimizationExperience,
    SelfOptimizationResult,
)
from tnfr.engines.self_optimization import TNFRSelfOptimizingEngine
from tnfr.utils.io import json_loads


@pytest.mark.parametrize(
    "recommendation,requested",
    [
        ("nodal_vec", OptimizationStrategy.NODAL_VECTORIZED),
        ("spectral_fft", OptimizationStrategy.SPECTRAL_FFT),
        ("use_spectral_methods", OptimizationStrategy.SPECTRAL_FFT),
        ("inspect_spectral_diagnostic", OptimizationStrategy.AUTO),
    ],
)
def test_automatic_dispatch_uses_registered_tokens_and_records_executed_service(
    recommendation, requested, monkeypatch
):
    import tnfr.dynamics.self_optimizing_engine as implementation

    monkeypatch.setattr(implementation, "HAS_CONSERVATION", False)
    engine = object.__new__(TNFRSelfOptimizingEngine)
    recommendations = SimpleNamespace(
        recommended_strategies=[recommendation], mathematical_insights={}
    )
    engine.recommend_optimization_strategy = lambda *_: recommendations
    admitted_nodes = []
    engine._prepare_sequence_validation = lambda *_, **kw: admitted_nodes.append(
        kw["node"]
    )
    actual = (
        OptimizationStrategy.STRUCTURAL_MEMO
        if requested is OptimizationStrategy.AUTO
        else requested
    )
    dispatched = []

    def execute(graph, operation, strategy, **kwargs):
        dispatched.append((strategy, kwargs))
        return OptimizationResult(
            strategy_used=actual,
            execution_time=0.0,
            speedup_factor=1.0,
            cache_hits=0,
            cache_misses=0,
            memory_used_mb=0.0,
            accuracy_preserved=True,
            details={"performance_measurements": {}},
        )

    engine.orchestrator = SimpleNamespace(
        analyze_optimization_profile=lambda *_: SimpleNamespace(edge_density=0.0),
        execute_optimization=execute,
    )
    experiences = []
    engine.learn_from_experience = experiences.append
    result = engine.optimize_automatically(nx.path_graph(2), node=0, node_id=1)
    assert dispatched == [(requested, {})]
    assert admitted_nodes == [0]
    assert result["requested_strategy"] == requested.value
    assert result["strategy_used"] == experiences[0].strategy_used == actual.value


def _read_signature(signature_path: Path) -> str:
    text = signature_path.read_text(encoding="utf-8").strip()
    # Signature lines follow the conventional "hash  filename" format.
    return text.split()[0]


def test_optimize_automatically_dry_run_creates_snapshot(tmp_path: Path) -> None:
    graph = nx.Graph()
    graph.add_edge(1, 2)

    engine = TNFRSelfOptimizingEngine()

    result = engine.optimize_automatically(
        graph,
        "diagnostic",
        dry_run=True,
        seed=42,
        node="alpha beta",
        operator_sequence=["AL", "UM", "IL", "SHA"],
        output_dir=tmp_path,
    )

    assert result["dry_run"] is True
    assert result["learning_updated"] is False
    assert result["telemetry_snapshots"] is not None

    snapshot_path = Path(result["snapshot_path"])
    assert snapshot_path.exists()

    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["operation_type"] == "diagnostic"
    assert payload["metadata"]["seed"] == "42"
    assert payload["metadata"]["node"].startswith("alpha")
    assert payload["validation"]["passed"] is True

    signature_path = snapshot_path.with_suffix(snapshot_path.suffix + ".sha256")
    assert signature_path.exists()
    assert result["signature"] == _read_signature(signature_path)
    assert result["signature"] == hashlib.sha256(snapshot_path.read_bytes()).hexdigest()


@pytest.mark.parametrize("node_count", (1, 2))
def test_snapshot_coherence_length_preserves_estimator_availability(node_count):
    from tnfr.physics.fields import (
        compute_unified_telemetry,
        estimate_coherence_length_with_provenance,
    )

    graph = nx.path_graph(node_count)
    for node in graph:
        graph.nodes[node].update(EPI=1.0, theta=0.0, nu_f=1.0, delta_nfr=0.0)
    expected = estimate_coherence_length_with_provenance(graph)
    raw = compute_unified_telemetry(graph)
    engine = object.__new__(TNFRSelfOptimizingEngine)
    snapshot = engine._capture_structural_snapshot(graph)
    canonical = snapshot["telemetry"]["canonical"]

    assert canonical["xi_c_provenance"]["method"] == expected.method
    assert canonical["xi_c_provenance"]["fit_quality"] == expected.fit_quality
    assert canonical["xi_c_available"] is (node_count == 2)
    if node_count == 1:
        assert math.isnan(raw["canonical"]["xi_c"])
        assert canonical["xi_c"] is None
    else:
        assert canonical["xi_c"] == expected.value == raw["canonical"]["xi_c"]
    assert canonical["phi_s"] == {str(node): 0.0 for node in graph}
    assert json_loads(json.dumps(snapshot, allow_nan=False)) == snapshot


def test_invalid_primitive_is_a_failed_snapshot_not_an_unavailable_estimator():
    graph = nx.path_graph(2)
    graph.nodes[0]["theta"] = True
    engine = object.__new__(TNFRSelfOptimizingEngine)
    snapshot = engine._capture_structural_snapshot(graph)
    assert snapshot["telemetry"] is None
    assert snapshot["telemetry_error"]
    assert graph.nodes[0]["theta"] is True


@pytest.mark.parametrize("field", ("xi_c", "phi_s"))
def test_unexpected_nonfinite_diagnostic_is_not_silently_projected(monkeypatch, field):
    import tnfr.dynamics.self_optimizing_engine as implementation
    from tnfr.physics.fields import compute_unified_telemetry

    graph = nx.path_graph(2)
    actual = compute_unified_telemetry(graph)
    if field == "xi_c":
        actual["canonical"][field] = math.nan
    else:
        actual["canonical"][field][0] = math.nan
    monkeypatch.setattr(implementation, "compute_unified_telemetry", lambda _: actual)
    engine = object.__new__(TNFRSelfOptimizingEngine)
    snapshot = engine._capture_structural_snapshot(graph)
    assert snapshot["telemetry"] is None
    assert snapshot["telemetry_error"]
    assert json_loads(json.dumps(snapshot, allow_nan=False)) == snapshot


@pytest.mark.parametrize("invalid", (math.nan, math.inf, object()))
def test_dry_run_serialization_failure_preserves_payload_and_signature(
    tmp_path, invalid
):
    directory = tmp_path / "seed"
    directory.mkdir()
    payload_path = directory / "node.json"
    signature_path = directory / "node.json.sha256"
    payload_path.write_bytes(b'{"retained":true}\n')
    signature_path.write_bytes(b"retained signature\n")
    before = {path: path.read_bytes() for path in directory.iterdir()}
    engine = object.__new__(TNFRSelfOptimizingEngine)
    with pytest.raises((TypeError, ValueError)):
        engine._persist_dry_run_payload(
            {"nested": {"value": invalid}}, tmp_path, "seed", "node"
        )
    assert {path: path.read_bytes() for path in directory.iterdir()} == before


def test_dry_run_serialization_failure_creates_no_output_directory(tmp_path):
    engine = object.__new__(TNFRSelfOptimizingEngine)
    with pytest.raises(ValueError):
        engine._persist_dry_run_payload(
            {"value": math.nan}, tmp_path / "not-created", "seed", "node"
        )
    assert list(tmp_path.iterdir()) == []


def test_optimize_automatically_dry_run_requires_valid_sequence(tmp_path: Path) -> None:
    graph = nx.Graph()
    graph.add_node(1)

    engine = TNFRSelfOptimizingEngine()

    with pytest.raises(ValueError):
        engine.optimize_automatically(
            graph,
            "diagnostic",
            dry_run=True,
            operator_sequence=["UM"],  # coupling without generator violates grammar
            output_dir=tmp_path,
        )


# ═══════════════════════════════════════════════════════════════════════════
# P5: Conservation-aware self-optimization
# ═══════════════════════════════════════════════════════════════════════════


def _make_tnfr_graph(n: int = 10) -> nx.Graph:
    """Build a small TNFR graph with proper node attributes."""
    G = nx.cycle_graph(n)
    for node in G.nodes():
        G.nodes[node]["EPI"] = 1.0 + 0.1 * node
        G.nodes[node]["nu_f"] = 1.0
        G.nodes[node]["ΔNFR"] = 0.05
        G.nodes[node]["theta"] = float(node) * 0.5
        G.nodes[node]["delta_nfr"] = 0.05
        G.nodes[node]["phase"] = float(node) * 0.5
    return G


class TestConservationFeedbackInResult:
    """SelfOptimizationResult carries conservation_feedback from the
    integrity monitor when available (P5: closed-loop data flow)."""

    def test_conservation_feedback_populated(self) -> None:
        """recommend_optimization_strategy() populates conservation_feedback."""
        G = _make_tnfr_graph()
        engine = TNFRSelfOptimizingEngine()
        result = engine.recommend_optimization_strategy(G, "general")
        assert isinstance(result, SelfOptimizationResult)
        # conservation_feedback is Optional — may be None if no monitor
        # is attached, but the field must exist on the dataclass.
        assert hasattr(result, "conservation_feedback")
        assert hasattr(result, "balance_feedback")

    def test_conservation_feedback_none_without_monitor(self) -> None:
        """Without an attached integrity monitor, conservation_feedback is None."""
        G = nx.Graph()
        G.add_edge(0, 1)
        engine = TNFRSelfOptimizingEngine()
        result = engine.recommend_optimization_strategy(G, "general")
        # No monitor attached → feedback_vector() returns nothing → None
        assert result.conservation_feedback is None
        assert result.balance_feedback is None

    def test_conservation_feedback_propagated_from_monitor(self) -> None:
        """When the integrity monitor is attached, conservation_feedback
        contains accurate diagnostic names and compatibility aliases."""
        from tnfr.physics.integrity import MonitorMode, enable_integrity_monitor

        G = _make_tnfr_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        # Force at least one data point into the monitor
        monitor._summary.total_operators = 1
        monitor._summary.balance_samples = 1
        monitor._summary.mean_conservation_quality = 0.85
        monitor._summary.mean_energy_derivative = -0.01
        monitor._summary.total_charge_drift = 0.02
        monitor._summary.violations_count = 0

        engine = TNFRSelfOptimizingEngine()
        result = engine.recommend_optimization_strategy(G, "general")
        cf = result.conservation_feedback
        assert cf is not None
        assert result.balance_feedback is cf
        assert result.mathematical_insights["balance_feedback"] is cf
        assert result.mathematical_insights["conservation_feedback"] is cf
        assert "balance_quality" in cf
        assert "candidate_energy_derivative" in cf
        assert "mean_structural_charge_drift" in cf
        assert "total_structural_charge_drift" in cf
        assert "structural_charge_drift" in cf
        assert "monitor_alert_rate" in cf
        assert "conservation_quality" in cf
        assert "energy_derivative" in cf
        assert "charge_drift" in cf
        assert "violation_rate" in cf
        assert cf["balance_quality"] == cf["conservation_quality"]
        assert cf["conservation_quality"] == pytest.approx(0.85)


class TestBalanceAlertsDoNotPrescribeStrategies:
    """Finite diagnostic alerts are exposed without prescribing operators."""

    def test_low_balance_quality_generates_review_only(self) -> None:
        from tnfr.physics.integrity import MonitorMode, enable_integrity_monitor

        G = _make_tnfr_graph(20)
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        # Simulate low conservation quality
        monitor._summary.total_operators = 10
        monitor._summary.balance_samples = 10
        monitor._summary.mean_conservation_quality = 0.4
        monitor._summary.mean_energy_derivative = 0.05
        monitor._summary.total_charge_drift = 0.5
        monitor._summary.violations_count = 3

        engine = TNFRSelfOptimizingEngine()
        result = engine.recommend_optimization_strategy(G, "general")

        cf = result.conservation_feedback
        assert cf is not None
        assert cf["balance_quality"] < 0.7
        strategies = result.recommended_strategies
        reviews = result.balance_alert_reviews
        assert "balance_quality_low_review" in reviews
        assert "candidate_energy_increase_review" in reviews
        assert "balance_quality_low_review" not in strategies
        assert "candidate_energy_increase_review" not in strategies
        assert "conservation_quality_low_stabilize" not in strategies
        assert "lyapunov_unstable_add_IL" not in strategies

    def test_no_balance_review_when_sample_is_within_alerts(self) -> None:
        from tnfr.physics.integrity import MonitorMode, enable_integrity_monitor

        G = _make_tnfr_graph(20)
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor._summary.total_operators = 10
        monitor._summary.balance_samples = 10
        monitor._summary.mean_conservation_quality = 0.95
        monitor._summary.mean_energy_derivative = -0.01
        monitor._summary.total_charge_drift = 0.01
        monitor._summary.violations_count = 0

        engine = TNFRSelfOptimizingEngine()
        result = engine.recommend_optimization_strategy(G, "general")
        cf = result.conservation_feedback
        assert cf is not None
        assert cf["balance_quality"] >= 0.7
        assert "balance_quality_low_review" not in result.balance_alert_reviews
        assert "candidate_energy_increase_review" not in result.balance_alert_reviews

    def test_attached_monitor_without_samples_does_not_generate_review(self) -> None:
        from tnfr.physics.integrity import MonitorMode, enable_integrity_monitor

        G = _make_tnfr_graph(20)
        enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)

        result = TNFRSelfOptimizingEngine().recommend_optimization_strategy(
            G, "general"
        )

        assert result.balance_feedback is not None
        assert result.balance_feedback["balance_sample_count"] == 0.0
        assert result.balance_alert_reviews == ()


class TestConservationInExperienceRecording:
    """Experience recording includes conservation metrics (P5)."""

    def test_experience_dataclass_accepts_conservation_fields(self) -> None:
        """OptimizationExperience.performance_metrics can hold conservation keys."""
        exp = OptimizationExperience(
            graph_properties={"nodes": 10, "edges": 15, "density": 0.3},
            operation_type="general",
            strategy_used="auto",
            parameters={},
            performance_metrics={
                "speedup_factor": 1.2,
                "execution_time": 0.05,
                "memory_used_mb": 10.0,
                "cache_hits": 5,
                "conservation_charge_drift": 0.02,
                "conservation_energy_derivative": -0.005,
                "conservation_rms_residual": 0.01,
            },
            timestamp=0.0,
            success=True,
        )
        assert exp.performance_metrics["conservation_charge_drift"] == pytest.approx(
            0.02
        )
        assert exp.performance_metrics[
            "conservation_energy_derivative"
        ] == pytest.approx(-0.005)


class TestAdaptiveConfigTracksConservation:
    """Adaptive configuration tracks the finite structural-charge diagnostic."""

    def test_mean_structural_charge_drift_tracked(self) -> None:
        """The accurate drift key is tracked with its compatibility alias."""
        engine = TNFRSelfOptimizingEngine()
        # Inject enough experiences to trigger learning (need >= 10)
        for i in range(12):
            exp = OptimizationExperience(
                graph_properties={"nodes": 10, "edges": 15, "density": 0.3},
                operation_type="general",
                strategy_used="auto",
                parameters={"backend": "numpy"},
                performance_metrics={
                    "speedup_factor": 1.1,
                    "execution_time": 0.05,
                    "memory_used_mb": 10.0,
                    "cache_hits": 5,
                    "conservation_charge_drift": 0.01 + 0.001 * i,
                    "conservation_energy_derivative": -0.003,
                    "conservation_rms_residual": 0.005,
                },
                timestamp=float(i),
                success=True,
            )
            engine.learn_from_experience(exp)

        assert "mean_structural_charge_drift" in engine.adaptive_config
        assert "mean_conservation_drift" in engine.adaptive_config
        assert engine.adaptive_config["mean_structural_charge_drift"] > 0
        assert engine.adaptive_config["mean_conservation_drift"] == pytest.approx(
            engine.adaptive_config["mean_structural_charge_drift"]
        )

    def test_no_conservation_drift_without_data(self) -> None:
        """If experiences lack conservation data, key is absent."""
        engine = TNFRSelfOptimizingEngine()
        for i in range(12):
            exp = OptimizationExperience(
                graph_properties={"nodes": 10, "edges": 15, "density": 0.3},
                operation_type="general",
                strategy_used="auto",
                parameters={"backend": "numpy"},
                performance_metrics={
                    "speedup_factor": 1.1,
                    "execution_time": 0.05,
                    "memory_used_mb": 10.0,
                    "cache_hits": 5,
                },
                timestamp=float(i),
                success=True,
            )
            engine.learn_from_experience(exp)

        assert "mean_conservation_drift" not in engine.adaptive_config
        assert "mean_structural_charge_drift" not in engine.adaptive_config


class TestConservationLowQualityRecommendations:
    """Scoped review recommendations are generated from monitor alerts."""

    def test_low_quality_generates_review_recommendation(self) -> None:
        from tnfr.physics.integrity import MonitorMode, enable_integrity_monitor

        G = _make_tnfr_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor._summary.total_operators = 5
        monitor._summary.balance_samples = 5
        monitor._summary.mean_conservation_quality = 0.3
        monitor._summary.mean_energy_derivative = 0.1
        monitor._summary.total_charge_drift = 0.5
        monitor._summary.violations_count = 2

        engine = TNFRSelfOptimizingEngine()
        result = engine.recommend_optimization_strategy(G, "general")
        strategies = result.recommended_strategies
        reviews = result.balance_alert_reviews

        assert "balance_quality_low_review" in reviews
        assert "candidate_energy_increase_review" in reviews
        assert "balance_quality_low_review" not in strategies
        assert "candidate_energy_increase_review" not in strategies
        assert "conservation_quality_low_stabilize" not in strategies
        assert "lyapunov_unstable_add_IL" not in strategies

    def test_high_structural_charge_drift_recommendation(self) -> None:
        from tnfr.physics.integrity import MonitorMode, enable_integrity_monitor

        G = _make_tnfr_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor._summary.total_operators = 5
        monitor._summary.balance_samples = 5
        monitor._summary.mean_conservation_quality = 0.9
        monitor._summary.mean_energy_derivative = -0.01
        monitor._summary.total_charge_drift = 1.0  # mean 0.2 across 5 samples
        monitor._summary.violations_count = 0

        engine = TNFRSelfOptimizingEngine()
        result = engine.recommend_optimization_strategy(G, "general")
        assert "structural_charge_drift_review" in result.balance_alert_reviews
        assert "structural_charge_drift_review" not in result.recommended_strategies

    def test_high_monitor_alert_rate_recommendation(self) -> None:
        from tnfr.physics.integrity import MonitorMode, enable_integrity_monitor

        G = _make_tnfr_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor._summary.total_operators = 10
        monitor._summary.balance_samples = 10
        monitor._summary.mean_conservation_quality = 0.9
        monitor._summary.mean_energy_derivative = -0.01
        monitor._summary.total_charge_drift = 0.01
        monitor._summary.violations_count = 5  # 50% violation rate

        engine = TNFRSelfOptimizingEngine()
        result = engine.recommend_optimization_strategy(G, "general")
        assert "monitor_alert_rate_review" in result.balance_alert_reviews
        assert "monitor_alert_rate_review" not in result.recommended_strategies
