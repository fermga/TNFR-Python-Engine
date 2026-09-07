"""Tests for TNFR self-optimization dry-run support."""

import json
from pathlib import Path

import networkx as nx
import pytest

from tnfr.dynamics.self_optimizing_engine import (
    OptimizationExperience,
    SelfOptimizationResult,
)
from tnfr.engines.self_optimization import TNFRSelfOptimizingEngine


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
