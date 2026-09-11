"""Evidence boundaries for advisory optimization and integration reports."""

from __future__ import annotations

import threading

import networkx as nx

from tnfr.dynamics.emergent_integration_engine import (
    IntegrationOpportunity,
    IntegrationPattern,
    IntegrationResult,
    TNFREmergentIntegrationEngine,
)
from tnfr.physics import fields


class _Objective:
    BALANCE_ALL = object()


class _AnalysisEngine:
    def __init__(self, *args, **kwargs):
        pass

    def analyze_mathematical_optimization_landscape(self, graph, operation):
        return {"scope": operation}


class _AutomaticEngine:
    def __init__(self, *args, **kwargs):
        raise AssertionError("advisory field analysis must not execute an optimizer")


def test_field_analysis_keeps_unmeasured_performance_values_neutral(monkeypatch):
    telemetry = {
        "complex_field": {"correlation": 0.99},
        "emergent_fields": {"chirality_magnitude": 0.75},
        "tensor_invariants": {"energy_density": [2.0]},
    }
    monkeypatch.setattr(fields, "_SELF_OPTIMIZING_AVAILABLE", True)
    monkeypatch.setattr(fields, "OptimizationObjective", _Objective)
    monkeypatch.setattr(fields, "compute_unified_telemetry", lambda graph: telemetry)
    monkeypatch.setattr(fields, "TNFRSelfOptimizingEngine", _AnalysisEngine)

    analysis = fields.analyze_optimization_potential(object())

    assert analysis["performance_evidence"] == "not_measured"
    assert analysis["predicted_improvements"] == {
        "field_correlation_speedup": None,
        "chirality_memory_reduction": None,
        "energy_computation_factor": None,
    }


def test_auto_field_path_does_not_claim_an_unmeasured_optimization(monkeypatch):
    telemetry = {"canonical": {"total_coherence": 1.0}}
    monkeypatch.setattr(fields, "_SELF_OPTIMIZING_AVAILABLE", True)
    monkeypatch.setattr(fields, "OptimizationObjective", _Objective)
    monkeypatch.setattr(fields, "TNFRSelfOptimizingEngine", _AutomaticEngine)
    monkeypatch.setattr(
        fields,
        "recommend_field_optimization_strategy",
        lambda graph, operation: {"unified_field_analysis": telemetry},
    )
    monkeypatch.setattr(
        fields,
        "compute_unified_telemetry",
        lambda graph: (_ for _ in ()).throw(AssertionError("duplicate computation")),
    )

    result = fields.auto_optimize_field_computation(object())

    assert result["result"] is telemetry
    assert result["optimization_applied"] is False
    assert result["performance_improvement"] is None
    assert result["performance_evidence"] == "not_measured"


def test_integration_forecasts_require_explicit_evidence():
    unmeasured = IntegrationPattern(
        pattern_id="p",
        opportunity_type=IntegrationOpportunity.SPECTRAL_SHARING,
        mathematical_basis="shared operator",
        involved_engines={"a", "b"},
        integration_strategy={},
        expected_benefit={"speedup": 4.0, "memory_savings": 0.5},
        mathematical_requirements=[],
        confidence_score=0.8,
    )
    measured = IntegrationPattern(
        pattern_id="m",
        opportunity_type=IntegrationOpportunity.SPECTRAL_SHARING,
        mathematical_basis="measured benchmark",
        involved_engines={"a", "b"},
        integration_strategy={},
        expected_benefit={"speedup": 1.2},
        mathematical_requirements=[],
        confidence_score=0.8,
        benefit_evidence="measured",
    )

    assert unmeasured.expected_benefit == {
        "speedup": None,
        "memory_savings": None,
    }
    assert measured.expected_benefit == {"speedup": 1.2}
    assert TNFREmergentIntegrationEngine._measure_baseline_performance(
        object(), object(), unmeasured
    ) == {}


def _pattern(opportunity_type: IntegrationOpportunity) -> IntegrationPattern:
    return IntegrationPattern(
        pattern_id=opportunity_type.value,
        opportunity_type=opportunity_type,
        mathematical_basis="contract test",
        involved_engines=set(),
        integration_strategy={},
        expected_benefit={},
        mathematical_requirements=[],
        confidence_score=0.5,
    )


def test_unimplemented_integration_routes_abstain_instead_of_claiming_success():
    engine = object.__new__(TNFREmergentIntegrationEngine)

    for opportunity_type, method_name in (
        (IntegrationOpportunity.VECTORIZATION_FUSION, "_apply_vectorization_fusion"),
        (IntegrationOpportunity.TEMPORAL_PREDICTION, "_apply_temporal_prediction"),
        (
            IntegrationOpportunity.PHASE_INFORMED_CACHING,
            "_apply_phase_informed_caching",
        ),
    ):
        success, details = getattr(engine, method_name)(
            _pattern(opportunity_type), object()
        )
        assert success is False
        assert details["application_status"] == "unavailable"
        assert details["performance_evidence"] == "not_measured"


def test_only_finite_metrics_with_measured_provenance_are_reported():
    metrics = {
        "speedup": 1.25,
        "boolean_is_not_a_measurement": True,
        "infinite": float("inf"),
        "text": "1.5",
    }

    assert TNFREmergentIntegrationEngine._measured_metric_mapping(
        metrics, "not_measured"
    ) == {}
    assert TNFREmergentIntegrationEngine._measured_metric_mapping(
        metrics, "measured"
    ) == {"speedup": 1.25}


def test_structural_validation_compares_actual_canonical_state():
    graph = nx.path_graph(2)
    for node in graph:
        graph.nodes[node].update(
            EPI=float(node),
            nu_f=1.0,
            phase=0.1 * node,
            delta_nfr=0.0,
            dEPI=0.0,
        )
    engine = object.__new__(TNFREmergentIntegrationEngine)
    pattern = _pattern(IntegrationOpportunity.SPECTRAL_SHARING)
    reference = engine._structural_state_signature(graph)

    assert reference is not None
    assert (
        engine._validate_mathematical_consistency(
            graph, pattern, reference_signature=reference
        )
        is True
    )

    graph.nodes[0]["EPI"] = 2.0
    assert (
        engine._validate_mathematical_consistency(
            graph, pattern, reference_signature=reference
        )
        is False
    )


def test_spectral_sharing_requires_an_authenticated_basis_signature():
    class _UnsignedFusion:
        @staticmethod
        def compute_structural_fields(graph, force_recompute=False):
            return {"field": graph, "force": force_recompute}

    engine = object.__new__(TNFREmergentIntegrationEngine)
    engine.spectral_fusion = _UnsignedFusion()
    success, details = engine._apply_spectral_sharing(
        _pattern(IntegrationOpportunity.SPECTRAL_SHARING), object()
    )

    assert success is False
    assert details["application_status"] == "abstained"
    assert details["reason"] == "authenticated_spectral_basis_not_materialized"


def _bare_integration_engine() -> TNFREmergentIntegrationEngine:
    engine = object.__new__(TNFREmergentIntegrationEngine)
    engine._lock = threading.RLock()
    engine.applied_integrations = []
    engine.discovered_patterns = {}
    engine.integration_opportunities = []
    for name in (
        "cache_orchestrator",
        "optimization_orchestrator",
        "self_optimizer",
        "spectral_fusion",
        "centralization",
        "nodal_optimizer",
        "structural_cache",
        "fft_cache",
    ):
        setattr(engine, name, None)
    return engine


def test_application_hides_unmeasured_metrics_even_from_details():
    engine = _bare_integration_engine()
    engine._apply_spectral_sharing = lambda pattern, graph: (
        True,
        {
            "application_status": "applied",
            "performance_evidence": "not_measured",
            "performance_improvement": {"speedup": 9.0},
            "resource_savings": {"memory_mb": 512.0},
        },
    )

    result = engine.apply_integration_pattern(
        _pattern(IntegrationOpportunity.SPECTRAL_SHARING),
        nx.path_graph(2),
        validate_mathematics=False,
    )

    assert result.success is True
    assert result.performance_evidence == "not_measured"
    assert result.performance_improvement == {}
    assert result.resource_savings == {}
    assert result.details["performance_improvement"] == {}
    assert result.details["resource_savings"] == {}


def test_statistics_do_not_hide_failed_validation_or_mix_metric_names():
    engine = _bare_integration_engine()
    engine.applied_integrations = [
        IntegrationResult(
            pattern_applied="speed",
            success=True,
            performance_improvement={"speedup": 1.2},
            mathematical_consistency_maintained=True,
            resource_savings={},
            side_effects=[],
            timestamp=1.0,
        ),
        IntegrationResult(
            pattern_applied="memory",
            success=True,
            performance_improvement={"memory_ratio": 0.8},
            mathematical_consistency_maintained=None,
            resource_savings={},
            side_effects=[],
            timestamp=2.0,
        ),
        IntegrationResult(
            pattern_applied="invalid",
            success=False,
            performance_improvement={},
            mathematical_consistency_maintained=False,
            resource_savings={},
            side_effects=[],
            timestamp=3.0,
        ),
    ]

    statistics = engine.get_integration_statistics()

    assert statistics["average_performance_improvement"] is None
    assert statistics["mean_performance_improvement_by_metric"] == {
        "speedup": 1.2,
        "memory_ratio": 0.8,
    }
    assert statistics["mathematical_consistency_rate"] == 0.5
    assert statistics["validation_evidence"] == "assessed"
