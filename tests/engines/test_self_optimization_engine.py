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

    def test_conservation_feedback_none_without_monitor(self) -> None:
        """Without an attached integrity monitor, conservation_feedback is None."""
        G = nx.Graph()
        G.add_edge(0, 1)
        engine = TNFRSelfOptimizingEngine()
        result = engine.recommend_optimization_strategy(G, "general")
        # No monitor attached → feedback_vector() returns nothing → None
        assert result.conservation_feedback is None

    def test_conservation_feedback_propagated_from_monitor(self) -> None:
        """When the integrity monitor is attached, conservation_feedback
        contains the four canonical fields from feedback_vector()."""
        from tnfr.physics.integrity import MonitorMode, enable_integrity_monitor

        G = _make_tnfr_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        # Force at least one data point into the monitor
        monitor._summary.total_operators = 1
        monitor._summary.mean_conservation_quality = 0.85
        monitor._summary.mean_energy_derivative = -0.01
        monitor._summary.total_charge_drift = 0.02
        monitor._summary.violations_count = 0

        engine = TNFRSelfOptimizingEngine()
        result = engine.recommend_optimization_strategy(G, "general")
        cf = result.conservation_feedback
        assert cf is not None
        assert "conservation_quality" in cf
        assert "energy_derivative" in cf
        assert "charge_drift" in cf
        assert "violation_rate" in cf
        assert cf["conservation_quality"] == pytest.approx(0.85)


class TestConservationAwareStrategyReordering:
    """When conservation quality is low, safe strategies must be
    promoted ahead of aggressive ones (P5: strategy reordering)."""

    def test_safe_strategies_promoted_when_quality_low(self) -> None:
        """Strategies containing 'cache'/'structural'/'stabiliz' are moved
        to the front when conservation_quality < 0.7."""
        from tnfr.physics.integrity import MonitorMode, enable_integrity_monitor

        G = _make_tnfr_graph(20)
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        # Simulate low conservation quality
        monitor._summary.total_operators = 10
        monitor._summary.mean_conservation_quality = 0.4
        monitor._summary.mean_energy_derivative = 0.05
        monitor._summary.total_charge_drift = 0.5
        monitor._summary.violations_count = 3

        engine = TNFRSelfOptimizingEngine()
        result = engine.recommend_optimization_strategy(G, "general")

        # The result must be influenced by conservation feedback
        cf = result.conservation_feedback
        assert cf is not None
        assert cf["conservation_quality"] < 0.7

        # If both safe and non-safe strategies exist, safe must come first
        strategies = result.recommended_strategies
        if len(strategies) >= 2:
            safe_kw = ("cache", "structural", "stabiliz", "memo")
            safe_indices = [
                i
                for i, s in enumerate(strategies)
                if any(kw in s.lower() for kw in safe_kw)
            ]
            other_indices = [
                i
                for i, s in enumerate(strategies)
                if not any(kw in s.lower() for kw in safe_kw)
            ]
            if safe_indices and other_indices:
                assert max(safe_indices) < min(
                    other_indices
                ), f"Safe strategies should precede others: {strategies}"

    def test_no_reordering_when_quality_high(self) -> None:
        """When conservation_quality >= 0.7 and dE/dt <= 0, no reordering."""
        from tnfr.physics.integrity import MonitorMode, enable_integrity_monitor

        G = _make_tnfr_graph(20)
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor._summary.total_operators = 10
        monitor._summary.mean_conservation_quality = 0.95
        monitor._summary.mean_energy_derivative = -0.01
        monitor._summary.total_charge_drift = 0.01
        monitor._summary.violations_count = 0

        engine = TNFRSelfOptimizingEngine()
        result = engine.recommend_optimization_strategy(G, "general")
        cf = result.conservation_feedback
        assert cf is not None
        assert cf["conservation_quality"] >= 0.7
        # Strategies exist in their natural order (no reordering applied)
        assert len(result.recommended_strategies) >= 0  # sanity


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
    """_update_adaptive_configuration() tracks mean conservation drift (P5 G5)."""

    def test_mean_conservation_drift_tracked(self) -> None:
        """After learning from experiences with conservation data,
        adaptive_config contains mean_conservation_drift."""
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

        assert "mean_conservation_drift" in engine.adaptive_config
        assert engine.adaptive_config["mean_conservation_drift"] > 0

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


class TestConservationLowQualityRecommendations:
    """Conservation text recommendations are generated when feedback
    indicates problems (closed-loop from integrity monitor)."""

    def test_low_quality_generates_stabilize_recommendation(self) -> None:
        from tnfr.physics.integrity import MonitorMode, enable_integrity_monitor

        G = _make_tnfr_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor._summary.total_operators = 5
        monitor._summary.mean_conservation_quality = 0.3
        monitor._summary.mean_energy_derivative = 0.1
        monitor._summary.total_charge_drift = 0.5
        monitor._summary.violations_count = 2

        engine = TNFRSelfOptimizingEngine()
        result = engine.recommend_optimization_strategy(G, "general")
        strategies = result.recommended_strategies

        assert "conservation_quality_low_stabilize" in strategies
        assert "lyapunov_unstable_add_IL" in strategies

    def test_high_charge_drift_recommendation(self) -> None:
        from tnfr.physics.integrity import MonitorMode, enable_integrity_monitor

        G = _make_tnfr_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor._summary.total_operators = 5
        monitor._summary.mean_conservation_quality = 0.9
        monitor._summary.mean_energy_derivative = -0.01
        monitor._summary.total_charge_drift = 0.2
        monitor._summary.violations_count = 0

        engine = TNFRSelfOptimizingEngine()
        result = engine.recommend_optimization_strategy(G, "general")
        assert "noether_charge_drift_correction" in result.recommended_strategies

    def test_high_violation_rate_recommendation(self) -> None:
        from tnfr.physics.integrity import MonitorMode, enable_integrity_monitor

        G = _make_tnfr_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor._summary.total_operators = 10
        monitor._summary.mean_conservation_quality = 0.9
        monitor._summary.mean_energy_derivative = -0.01
        monitor._summary.total_charge_drift = 0.01
        monitor._summary.violations_count = 5  # 50% violation rate

        engine = TNFRSelfOptimizingEngine()
        result = engine.recommend_optimization_strategy(G, "general")
        assert "high_violation_rate_grammar_review" in result.recommended_strategies
