"""Independent scalar-admission and reduction controls for optimization evidence."""

from __future__ import annotations

import math
from fractions import Fraction as Q

import networkx as nx
import numpy as np
import pytest

from tnfr.dynamics import self_optimizing_engine as implementation
from tnfr.dynamics.self_optimizing_engine import (
    OptimizationExperience,
    TNFRSelfOptimizingEngine,
)
from tnfr.errors import TNFRValueError


def _graph(forms=(0.0, 0.0)):
    graph = nx.path_graph(len(forms))
    for node, form in enumerate(forms):
        graph.nodes[node].update(EPI=form, nu_f=1, theta=0, delta_nfr=0)
    return graph


@pytest.fixture
def landscape(monkeypatch):
    """Isolate the nodal diagnostic from unrelated optional field analyses."""
    monkeypatch.setattr(implementation, "HAS_UNIFIED_FIELDS", False)
    engine = object.__new__(TNFRSelfOptimizingEngine)
    engine.pattern_engine = None
    return engine.analyze_mathematical_optimization_landscape


def test_large_offset_variance_matches_exact_two_sample_difference(landscape):
    left, right = 1e155, 1e155 + 1e140
    expected = float((Q(right) - Q(left)) ** 2 / 4)

    analysis = landscape(_graph((left, right)))["nodal_equation_analysis"]

    assert math.isfinite(expected) and expected > 0
    assert analysis["epi_variance"] == pytest.approx(expected, rel=2e-15)


@pytest.mark.parametrize("forms", ((1e308, -1e308), (1e-200, -1e-200)))
def test_unrepresentable_variance_does_not_become_zero_or_trigger_policy(
    landscape, forms
):
    analysis = landscape(_graph(forms))["nodal_equation_analysis"]

    assert analysis["epi_variance"] is None
    assert (
        "low_variance_epi_optimization" not in analysis["optimization_recommendations"]
    )


@pytest.mark.parametrize("forms,expected", (((-2.0, 1.0), 2.25), ((0.0, 0.0), 0.0)))
def test_ordinary_and_exact_zero_variance_remain_available(landscape, forms, expected):
    analysis = landscape(_graph(forms))["nodal_equation_analysis"]

    assert analysis["epi_variance"] == expected
    assert (
        "low_variance_epi_optimization" in analysis["optimization_recommendations"]
    ) is (expected == 0)


def test_large_pressure_mean_remains_finite_without_overflowing_sum(landscape):
    graph = _graph()
    for node in graph:
        graph.nodes[node]["delta_nfr"] = 1e308

    analysis = landscape(graph)["nodal_equation_analysis"]

    assert analysis["dnfr_magnitude"] == 1e308
    assert "high_dnfr_stabilization" in analysis["optimization_recommendations"]


@pytest.mark.parametrize("channel", ("nu_f", "delta_nfr"))
@pytest.mark.parametrize(
    "invalid", (Q(1, 10**400), Q(-1, 10**400), True, np.bool_(False), "1", math.inf)
)
def test_invalid_authoritative_nodal_values_cannot_become_zero(
    landscape, channel, invalid
):
    graph = _graph()
    graph.nodes[0][channel] = invalid
    graph.nodes[0]["vf" if channel == "nu_f" else "dnfr"] = 1

    with pytest.raises(TNFRValueError, match="finite"):
        landscape(graph)

    assert graph.nodes[0][channel] is invalid


def test_negative_capacity_rejects_but_signed_pressure_and_zero_capacity_are_valid(
    landscape,
):
    graph = _graph()
    graph.nodes[0]["nu_f"] = Q(-1, 4)
    with pytest.raises(TNFRValueError, match="nonnegative"):
        landscape(graph)

    graph.nodes[0]["nu_f"] = 0
    graph.nodes[0]["delta_nfr"] = -2
    analysis = landscape(graph)["nodal_equation_analysis"]
    assert analysis["vf_range"] == 1
    assert analysis["dnfr_magnitude"] == 1


@pytest.mark.parametrize("shape", ("mapping", "list", "object_array"))
@pytest.mark.parametrize("invalid", (True, Q(1, 10**400), math.nan))
def test_snapshot_sense_mean_admits_original_samples_before_array_conversion(
    monkeypatch, shape, invalid
):
    payload = {
        "mapping": {0: invalid},
        "list": [invalid],
        "object_array": np.array([invalid], dtype=object),
    }[shape]
    monkeypatch.setattr(implementation, "HAS_UNIFIED_FIELDS", False)
    monkeypatch.setattr(implementation, "compute_Si", lambda *_a, **_kw: payload)
    engine = object.__new__(TNFRSelfOptimizingEngine)

    snapshot = engine._capture_structural_snapshot(_graph())

    assert "sense_index" not in snapshot


def test_snapshot_sense_mean_keeps_representable_large_result(monkeypatch):
    monkeypatch.setattr(implementation, "HAS_UNIFIED_FIELDS", False)
    monkeypatch.setattr(implementation, "compute_Si", lambda *_a, **_kw: [1e308, 1e308])
    engine = object.__new__(TNFRSelfOptimizingEngine)

    assert engine._capture_structural_snapshot(_graph())["sense_index"] == 1e308


def test_learning_uses_finite_means_to_compare_large_measured_speedups():
    engine = TNFRSelfOptimizingEngine()
    for index in range(12):
        backend, speedup = ("numpy", 1e308) if index < 6 else ("faster", 1.5e308)
        engine.learn_from_experience(
            OptimizationExperience(
                graph_properties={"nodes": 2, "density": 1.0},
                operation_type="general",
                strategy_used=backend,
                parameters={"backend": backend},
                performance_metrics={
                    "speedup_factor": speedup,
                    "memory_used_mb": 1e308,
                    "structural_charge_drift": 1e308,
                },
                timestamp=index,
                success=True,
            )
        )

    assert engine.adaptive_config["backend_preference"] == "faster"
    assert engine.adaptive_config["mean_structural_charge_drift"] == 1e308
    assert engine.learned_policies[-1].average_improvement == 1.5e308
