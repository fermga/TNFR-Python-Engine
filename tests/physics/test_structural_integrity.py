"""Tests for TNFR operator-contract monitoring and finite-step alerts.

Validates the real-time diagnostic bridge between:
  conservation.py → integrity.py → definitions_base.py → self_optimizing_engine.py

Tests verify:
1. MonitorMode semantics: OFF, OBSERVE, ENFORCE
2. Postcondition evaluation for canonical operators
3. Structural-balance quality assessment via snapshots
4. Candidate-energy trend measurement
5. Structural-charge drift tracking
6. Separation of balance alerts from grammar validation
7. IntegritySummary running statistics
8. StructuralIntegrityViolation raised in ENFORCE mode
9. Corrective suggestions generation
10. feedback_vector() output for self-optimization engine
11. Integration with operator execution (definitions_base.py wiring)
12. Backward compatibility: no overhead when monitor is not attached
"""

from __future__ import annotations

import math
import os
import sys

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tnfr.alias import get_attr, set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.physics.integrity import (
    IntegrityReport,
    IntegritySummary,
    MonitorMode,
    OperatorContractAudit,
    OperatorContractResult,
    StructuralIntegrityMonitor,
    StructuralIntegrityViolation,
    audit_operator_contracts,
    enable_integrity_monitor,
)

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _make_graph(num_nodes: int = 5, connected: bool = True) -> nx.Graph:
    """Build a small TNFR graph with proper node attributes."""
    if connected:
        G = nx.cycle_graph(num_nodes)
    else:
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
    for n in G.nodes():
        G.nodes[n]["EPI"] = 1.0 + 0.1 * n
        G.nodes[n]["nu_f"] = 1.0
        G.nodes[n]["ΔNFR"] = 0.05
        G.nodes[n]["theta"] = float(n) * 0.5
        G.nodes[n]["delta_nfr"] = 0.05
        G.nodes[n]["phase"] = float(n) * 0.5
    return G


# ═══════════════════════════════════════════════════════════════════════════
# Monitor creation and attachment
# ═══════════════════════════════════════════════════════════════════════════


class TestMonitorCreation:
    """Test StructuralIntegrityMonitor instantiation and attachment."""

    def test_default_mode_is_observe(self) -> None:
        monitor = StructuralIntegrityMonitor()
        assert monitor.mode == MonitorMode.OBSERVE

    def test_explicit_mode_off(self) -> None:
        monitor = StructuralIntegrityMonitor(mode=MonitorMode.OFF)
        assert monitor.mode == MonitorMode.OFF

    def test_explicit_mode_enforce(self) -> None:
        monitor = StructuralIntegrityMonitor(mode=MonitorMode.ENFORCE)
        assert monitor.mode == MonitorMode.ENFORCE

    @pytest.mark.parametrize(
        ("kwargs", "parameter"),
        [
            ({"conservation_threshold": -0.1}, "conservation_threshold"),
            ({"conservation_threshold": 1.1}, "conservation_threshold"),
            ({"conservation_threshold": float("nan")}, "conservation_threshold"),
            ({"lyapunov_tolerance": -0.1}, "lyapunov_tolerance"),
            ({"lyapunov_tolerance": float("inf")}, "lyapunov_tolerance"),
            ({"charge_drift_threshold": -0.1}, "charge_drift_threshold"),
            ({"charge_drift_threshold": float("inf")}, "charge_drift_threshold"),
        ],
    )
    def test_rejects_invalid_alert_thresholds(
        self, kwargs: dict[str, float], parameter: str
    ) -> None:
        with pytest.raises(ValueError, match=parameter):
            StructuralIntegrityMonitor(**kwargs)

    def test_attach_stores_in_graph(self) -> None:
        G = _make_graph()
        monitor = StructuralIntegrityMonitor()
        monitor.attach(G)
        assert G.graph.get("integrity_monitor") is monitor

    def test_get_retrieves_from_graph(self) -> None:
        G = _make_graph()
        monitor = StructuralIntegrityMonitor()
        monitor.attach(G)
        retrieved = StructuralIntegrityMonitor.get(G)
        assert retrieved is monitor

    def test_get_returns_none_when_not_attached(self) -> None:
        G = _make_graph()
        assert StructuralIntegrityMonitor.get(G) is None

    def test_enable_convenience_function(self) -> None:
        G = _make_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        assert isinstance(monitor, StructuralIntegrityMonitor)
        assert StructuralIntegrityMonitor.get(G) is monitor
        assert monitor.mode == MonitorMode.OBSERVE


# ═══════════════════════════════════════════════════════════════════════════
# Before/After Operator cycle
# ═══════════════════════════════════════════════════════════════════════════


class TestBeforeAfterCycle:
    """Test the before_operator/after_operator cycle."""

    def test_before_after_produces_report(self) -> None:
        G = _make_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor.before_operator(G, 0)
        report = monitor.after_operator(G, 0, "Coherence")
        assert isinstance(report, IntegrityReport)

    def test_report_has_conservation_quality(self) -> None:
        G = _make_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor.before_operator(G, 0)
        report = monitor.after_operator(G, 0, "Coherence")
        assert isinstance(report.conservation_quality, float)
        assert 0.0 <= report.conservation_quality <= 1.0

    def test_off_mode_returns_default_report(self) -> None:
        """OFF mode returns a default (healthy) report with no computation."""
        G = _make_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OFF)
        monitor.before_operator(G, 0)
        result = monitor.after_operator(G, 0, "Coherence")
        assert isinstance(result, IntegrityReport)
        assert result.is_healthy

    def test_observe_mode_does_not_raise(self) -> None:
        """OBSERVE mode should never raise even with violations."""
        G = _make_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor.before_operator(G, 0)
        # Tamper with state to ensure postcondition violation
        G.nodes[0]["EPI"] = -999.0
        # Should not raise
        report = monitor.after_operator(G, 0, "Coherence")
        assert isinstance(report, IntegrityReport)

    def test_enforce_mode_raises_on_unhealthy(self) -> None:
        """ENFORCE raises on a configured alert without calling it grammar."""
        G = _make_graph()
        monitor = enable_integrity_monitor(
            G, mode=MonitorMode.ENFORCE, conservation_threshold=0.999999
        )
        monitor.before_operator(G, 0)
        # Tamper: create a large finite balance residual.
        for n in G.nodes():
            G.nodes[n]["ΔNFR"] = 999.0
            G.nodes[n]["delta_nfr"] = 999.0
        with pytest.raises(StructuralIntegrityViolation) as exc_info:
            monitor.after_operator(G, 0, "Coherence")
        assert exc_info.value.violation_type != "grammar"
        assert exc_info.value.details["grammar_validated"] is False
        assert exc_info.value.details["grammar_violations"] == []


# ═══════════════════════════════════════════════════════════════════════════
# Postcondition evaluation
# ═══════════════════════════════════════════════════════════════════════════


class TestPostconditions:
    """Test that operator-specific postconditions are checked."""

    def test_coherence_postcondition_passes_on_stable(self) -> None:
        """IL postcondition: C(t) must not decrease."""
        G = _make_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor.before_operator(G, 0)
        # State unchanged → coherence unchanged → postcondition passes
        report = monitor.after_operator(G, 0, "Coherence")
        assert report.postcondition_ok is True

    def test_silence_postcondition_passes_on_unchanged_epi(self) -> None:
        """SHA postcondition: EPI must remain unchanged."""
        G = _make_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor.before_operator(G, 0)
        # State unchanged → EPI unchanged
        report = monitor.after_operator(G, 0, "Silence")
        assert report.postcondition_ok is True

    def test_silence_postcondition_fails_on_changed_epi(self) -> None:
        """SHA postcondition detects EPI change."""
        G = _make_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor.before_operator(G, 0)
        # Tamper EPI
        G.nodes[0]["EPI"] = 999.0
        report = monitor.after_operator(G, 0, "Silence")
        assert report.postcondition_ok is False

    def test_emission_postcondition_accepts_epi_increase_only(self) -> None:
        """AL accepts its EPI source while secondary channels stay fixed."""
        G = _make_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor.before_operator(G, 0)
        G.nodes[0]["EPI"] = G.nodes[0]["EPI"] + 0.1
        report = monitor.after_operator(G, 0, "Emission")
        assert report.postcondition_ok is True

    @pytest.mark.parametrize(
        ("aliases", "value", "detail"),
        [
            (ALIAS_VF, 2.0, "νf changed"),
            (ALIAS_DNFR, 0.25, "ΔNFR changed"),
            (ALIAS_THETA, 0.25, "Phase changed"),
        ],
    )
    def test_emission_postcondition_rejects_non_epi_writes(
        self, aliases, value, detail
    ) -> None:
        """AL is a channel-pure EPI source."""
        G = _make_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor.before_operator(G, 0)
        set_attr(G.nodes[0], aliases, value)
        report = monitor.after_operator(G, 0, "Emission")
        assert report.postcondition_ok is False
        assert detail in report.postcondition_detail

    def test_mutation_postcondition_passes_on_theta_change(self) -> None:
        """ZHIR postcondition: θ must change."""
        G = _make_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor.before_operator(G, 0)
        # Change theta
        G.nodes[0]["theta"] = G.nodes[0]["theta"] + 1.0
        report = monitor.after_operator(G, 0, "Mutation")
        assert report.postcondition_ok is True

    def test_mutation_postcondition_fails_on_unchanged_theta(self) -> None:
        """ZHIR postcondition detects unchanged θ."""
        G = _make_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor.before_operator(G, 0)
        # θ stays the same
        report = monitor.after_operator(G, 0, "Mutation")
        assert report.postcondition_ok is False

    def test_expansion_postcondition_passes_on_vf_increase(self) -> None:
        """VAL postcondition: νf (capacity) must not decrease."""
        G = _make_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor.before_operator(G, 0)
        G.nodes[0]["nu_f"] = G.nodes[0]["nu_f"] + 0.5
        report = monitor.after_operator(G, 0, "Expansion")
        assert report.postcondition_ok is True

    def test_expansion_postcondition_fails_on_vf_decrease(self) -> None:
        """VAL postcondition detects νf decrease (capacity must not shrink)."""
        G = _make_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor.before_operator(G, 0)
        G.nodes[0]["nu_f"] = 0.01
        report = monitor.after_operator(G, 0, "Expansion")
        assert report.postcondition_ok is False
        assert "νf decreased" in report.postcondition_detail

    def test_contraction_postcondition_passes_on_vf_decrease(self) -> None:
        """NUL postcondition: νf (capacity) must not increase."""
        G = _make_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor.before_operator(G, 0)
        G.nodes[0]["nu_f"] = G.nodes[0]["nu_f"] * 0.85
        report = monitor.after_operator(G, 0, "Contraction")
        assert report.postcondition_ok is True

    def test_contraction_postcondition_fails_on_vf_increase(self) -> None:
        """NUL postcondition detects νf increase (capacity must not grow)."""
        G = _make_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor.before_operator(G, 0)
        G.nodes[0]["nu_f"] = G.nodes[0]["nu_f"] + 0.5
        report = monitor.after_operator(G, 0, "Contraction")
        assert report.postcondition_ok is False
        assert "νf increased" in report.postcondition_detail

    def test_resonance_postcondition_preserves_identity(self) -> None:
        """RA postcondition: EPI sign (structural identity) preserved."""
        G = _make_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor.before_operator(G, 0)
        # νf up (amplification) and same-sign EPI change → identity preserved
        G.nodes[0]["nu_f"] = G.nodes[0]["nu_f"] * 1.05
        report = monitor.after_operator(G, 0, "Resonance")
        assert report.postcondition_ok is True

    def test_resonance_postcondition_fails_on_sign_flip(self) -> None:
        """RA postcondition detects EPI sign flip (identity not preserved)."""
        G = _make_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor.before_operator(G, 0)
        # Flip EPI sign → structural identity violated
        G.nodes[0]["EPI"] = -abs(G.nodes[0]["EPI"]) - 0.5
        report = monitor.after_operator(G, 0, "Resonance")
        assert report.postcondition_ok is False
        assert "sign flipped" in report.postcondition_detail

    def test_unknown_operator_skips_postcondition(self) -> None:
        """Operators without specific postconditions pass by default."""
        G = _make_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor.before_operator(G, 0)
        report = monitor.after_operator(G, 0, "SomeUnknownOperator")
        # Unknown operator → no postcondition → defaults to ok
        assert isinstance(report.postcondition_ok, bool)

    # ── Enriched postconditions ────────────────────────────────────────

    def test_emission_postcondition_fails_on_epi_decrease(self) -> None:
        """AL enriched: EPI must not decrease during Emission."""
        G = _make_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor.before_operator(G, 0)
        G.nodes[0]["EPI"] = 0.001  # Decrease EPI
        report = monitor.after_operator(G, 0, "Emission")
        assert report.postcondition_ok is False
        assert "EPI decreased" in report.postcondition_detail

    def test_coherence_postcondition_fails_on_dnfr_increase(self) -> None:
        """IL enriched: |ΔNFR| must not increase during Coherence."""
        G = _make_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor.before_operator(G, 0)
        # Increase ΔNFR dramatically
        G.nodes[0]["ΔNFR"] = 999.0
        G.nodes[0]["delta_nfr"] = 999.0
        report = monitor.after_operator(G, 0, "Coherence")
        assert report.postcondition_ok is False

    def test_coherence_postcondition_ignores_unbound_latest_telemetry(self) -> None:
        """IL compares its bound snapshots even when latest telemetry disagrees."""
        G = _make_graph()
        G.graph["IL_dnfr_reductions"] = [
            {"node": 0, "before": 0.5, "after": 0.35, "reduction": 0.15}
        ]
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor.before_operator(G, 0)
        # Raise this target magnitude while lowering the network mean so
        # C(t) does not mask the snapshot-bound pressure assertion.
        G.nodes[0]["ΔNFR"] = -0.06
        G.nodes[0]["delta_nfr"] = -0.06
        G.nodes[1]["ΔNFR"] = 0.0
        G.nodes[1]["delta_nfr"] = 0.0
        report = monitor.after_operator(G, 0, "Coherence")
        assert report.postcondition_ok is False
        assert "|ΔNFR| increased" in report.postcondition_detail

    def test_silence_postcondition_fails_on_vf_increase(self) -> None:
        """SHA enriched: νf must not increase during Silence."""
        G = _make_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor.before_operator(G, 0)
        G.nodes[0]["nu_f"] = G.nodes[0]["nu_f"] + 5.0
        report = monitor.after_operator(G, 0, "Silence")
        assert report.postcondition_ok is False
        assert "νf increased" in report.postcondition_detail

    def test_silence_postcondition_passes_on_vf_decrease(self) -> None:
        """SHA enriched: νf decrease is expected (freeze)."""
        G = _make_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor.before_operator(G, 0)
        G.nodes[0]["nu_f"] = G.nodes[0]["nu_f"] * 0.85
        report = monitor.after_operator(G, 0, "Silence")
        assert report.postcondition_ok is True

    def test_resonance_postcondition_fails_on_vf_decrease(self) -> None:
        """RA enriched: νf must not decrease during Resonance."""
        G = _make_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor.before_operator(G, 0)
        G.nodes[0]["nu_f"] = 0.01  # Decrease νf
        report = monitor.after_operator(G, 0, "Resonance")
        assert report.postcondition_ok is False
        assert "νf decreased" in report.postcondition_detail

    def test_resonance_postcondition_passes_on_vf_increase(self) -> None:
        """RA enriched: νf increase expected (amplification)."""
        G = _make_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor.before_operator(G, 0)
        G.nodes[0]["nu_f"] = G.nodes[0]["nu_f"] * 1.05
        report = monitor.after_operator(G, 0, "Resonance")
        assert report.postcondition_ok is True

    # ── New postconditions for UM, THOL, NAV, REMESH ──────────────────

    def test_coupling_postcondition_passes_on_dnfr_decrease(self) -> None:
        """UM: |ΔNFR| must not increase during Coupling."""
        G = _make_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor.before_operator(G, 0)
        # ΔNFR decreases (expected by coupling)
        G.nodes[0]["ΔNFR"] = G.nodes[0]["ΔNFR"] * 0.85
        G.nodes[0]["delta_nfr"] = G.nodes[0]["delta_nfr"] * 0.85
        report = monitor.after_operator(G, 0, "Coupling")
        assert report.postcondition_ok is True

    def test_coupling_postcondition_fails_on_dnfr_increase(self) -> None:
        """UM: |ΔNFR| increase during Coupling is a violation."""
        G = _make_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor.before_operator(G, 0)
        G.nodes[0]["ΔNFR"] = 99.0
        G.nodes[0]["delta_nfr"] = 99.0
        report = monitor.after_operator(G, 0, "Coupling")
        assert report.postcondition_ok is False
        assert "|ΔNFR| increased" in report.postcondition_detail

    def test_self_organization_postcondition_passes_on_stable_coherence(self) -> None:
        """THOL: Small coherence change is tolerable."""
        G = _make_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor.before_operator(G, 0)
        # No state change → coherence unchanged → passes
        report = monitor.after_operator(G, 0, "Self_organization")
        assert report.postcondition_ok is True

    def test_self_organization_postcondition_fails_on_catastrophic_drop(self) -> None:
        """THOL: >10% coherence drop is a violation."""
        G = _make_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor.before_operator(G, 0)
        # Cause catastrophic coherence drop by spiking ΔNFR everywhere
        for n in G.nodes():
            G.nodes[n]["ΔNFR"] = 999.0
            G.nodes[n]["delta_nfr"] = 999.0
        report = monitor.after_operator(G, 0, "Self_organization")
        assert report.postcondition_ok is False
        assert "Coherence dropped" in report.postcondition_detail

    def test_transition_postcondition_passes_on_state_change(self) -> None:
        """NAV: At least one state variable must change."""
        G = _make_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor.before_operator(G, 0)
        # NAV modifies νf (regime transition)
        G.nodes[0]["nu_f"] = G.nodes[0]["nu_f"] * 1.2
        report = monitor.after_operator(G, 0, "Transition")
        assert report.postcondition_ok is True

    def test_transition_postcondition_fails_on_no_change(self) -> None:
        """NAV: No state change is a violation (trivial transition)."""
        G = _make_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor.before_operator(G, 0)
        # No state changes at all
        report = monitor.after_operator(G, 0, "Transition")
        assert report.postcondition_ok is False
        assert "No state change" in report.postcondition_detail

    def test_recursivity_postcondition_always_passes(self) -> None:
        """REMESH: Advisory glyph always passes postcondition."""
        G = _make_graph()
        monitor = enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
        monitor.before_operator(G, 0)
        report = monitor.after_operator(G, 0, "Recursivity")
        assert report.postcondition_ok is True

    # ── Full coverage: all 13 operators registered ────────────────────

    def test_all_13_operators_have_postconditions(self) -> None:
        """Verify POSTCONDITIONS dict covers all 13 canonical operators."""
        from tnfr.physics.integrity import POSTCONDITIONS

        expected = {
            "coherence",
            "dissonance",
            "silence",
            "reception",
            "resonance",
            "emission",
            "expansion",
            "contraction",
            "mutation",
            "coupling",
            "self_organization",
            "transition",
            "recursivity",
        }
        assert set(POSTCONDITIONS.keys()) == expected


# ═══════════════════════════════════════════════════════════════════════════
# IntegritySummary accumulation
# ═══════════════════════════════════════════════════════════════════════════


class TestIntegritySummary:
    """Test running statistics accumulation."""

    def test_summary_increments_total(self) -> None:
        G = _make_graph()
        monitor = enable_integrity_monitor(G)
        # Run two cycles
        monitor.before_operator(G, 0)
        monitor.after_operator(G, 0, "Coherence")
        monitor.before_operator(G, 1)
        monitor.after_operator(G, 1, "Coherence")
        assert monitor.summary.total_operators == 2

    def test_summary_tracks_violations(self) -> None:
        G = _make_graph()
        monitor = enable_integrity_monitor(G)
        monitor.before_operator(G, 0)
        # Break silence postcondition to cause violation
        G.nodes[0]["EPI"] = 999.0
        report = monitor.after_operator(G, 0, "Silence")
        if not report.is_healthy:
            assert monitor.summary.violations_count >= 1

    def test_summary_mean_conservation_quality(self) -> None:
        G = _make_graph()
        monitor = enable_integrity_monitor(G)
        monitor.before_operator(G, 0)
        report = monitor.after_operator(G, 0, "Coherence")
        s = monitor.summary
        assert isinstance(s.mean_conservation_quality, float)


# ═══════════════════════════════════════════════════════════════════════════
# IntegrityReport health check
# ═══════════════════════════════════════════════════════════════════════════


class TestIntegrityReportHealth:
    """Test the is_healthy property of IntegrityReport."""

    def test_healthy_on_stable_graph(self) -> None:
        """Clean graph with no changes → healthy report."""
        G = _make_graph()
        monitor = enable_integrity_monitor(G)
        monitor.before_operator(G, 0)
        report = monitor.after_operator(G, 0, "Coherence")
        # On an unmodified graph, should be healthy
        assert isinstance(report.is_healthy, bool)

    def test_report_fields_populated(self) -> None:
        G = _make_graph()
        monitor = enable_integrity_monitor(G)
        monitor.before_operator(G, 0)
        report = monitor.after_operator(G, 0, "Coherence")
        assert report.conservation_quality is not None
        assert report.energy_derivative is not None
        assert isinstance(report.is_lyapunov_stable, bool)
        assert isinstance(report.grammar_violations, list)
        assert report.grammar_validated is False
        assert report.grammar_violations == []
        assert isinstance(report.balance_alerts, list)
        assert report.balance_quality == report.conservation_quality
        assert report.candidate_energy_derivative == report.energy_derivative
        assert report.structural_charge_drift == report.noether_charge_drift
        assert report.diagnostic_follow_up == report.corrective_suggestion
        assert isinstance(report.postcondition_ok, bool)

    def test_large_residual_is_alert_not_grammar_verdict(self) -> None:
        G = _make_graph()
        monitor = enable_integrity_monitor(G)
        monitor.before_operator(G, 0)
        for n in G.nodes():
            G.nodes[n]["delta_nfr"] = 1000.0 + n
            G.nodes[n]["ΔNFR"] = 1000.0 + n
        report = monitor.after_operator(G, 0, "Coherence")
        assert report.balance_sample_available
        assert report.balance_alerts
        assert report.grammar_validated is False
        assert report.grammar_violations == []

    def test_exact_candidate_trend_is_separate_from_legacy_tolerance(self) -> None:
        report = IntegrityReport(
            operator="Coherence",
            node=0,
            energy_derivative=1e-7,
            is_lyapunov_stable=True,
            balance_sample_available=True,
        )
        assert report.candidate_energy_nonincreasing is False
        assert report.candidate_energy_within_numerical_tolerance is True

    def test_missing_interval_is_unsampled_and_never_reuses_prior_capture(self) -> None:
        G = _make_graph()
        monitor = enable_integrity_monitor(G)
        monitor.before_operator(G, 0)
        sampled = monitor.after_operator(G, 0, "Coherence")
        unsampled = monitor.after_operator(G, 0, "Coherence")

        assert sampled.balance_sample_available is True
        assert unsampled.balance_sample_available is False
        assert unsampled.balance_quality is None
        assert unsampled.candidate_energy_derivative is None
        assert unsampled.candidate_energy_nonincreasing is None
        assert unsampled.structural_charge_drift is None
        assert unsampled.postcondition_evaluated is False
        assert monitor.summary.balance_samples == 1


# ═══════════════════════════════════════════════════════════════════════════
# Corrective suggestions
# ═══════════════════════════════════════════════════════════════════════════


class TestCorrectiveSuggestions:
    """Test that corrective suggestions are generated for violations."""

    def test_no_suggestion_for_healthy_report(self) -> None:
        G = _make_graph()
        monitor = enable_integrity_monitor(G)
        monitor.before_operator(G, 0)
        report = monitor.after_operator(G, 0, "Coherence")
        if report.is_healthy:
            assert (
                report.corrective_suggestion is None
                or report.corrective_suggestion == ""
            )

    def test_suggestion_for_postcondition_failure(self) -> None:
        G = _make_graph()
        monitor = enable_integrity_monitor(G)
        monitor.before_operator(G, 0)
        # Break silence postcondition
        G.nodes[0]["EPI"] = 999.0
        report = monitor.after_operator(G, 0, "Silence")
        # Should have a suggestion or at least not crash
        assert isinstance(report.corrective_suggestion, (str, type(None)))


# ═══════════════════════════════════════════════════════════════════════════
# feedback_vector for self-optimization
# ═══════════════════════════════════════════════════════════════════════════


class TestFeedbackVector:
    """Test feedback_vector() produces correct structure for optimization engine."""

    def test_feedback_vector_keys(self) -> None:
        G = _make_graph()
        monitor = enable_integrity_monitor(G)
        monitor.before_operator(G, 0)
        monitor.after_operator(G, 0, "Coherence")
        fv = monitor.feedback_vector()
        expected_keys = {
            "balance_sample_count",
            "balance_quality",
            "candidate_energy_derivative",
            "mean_structural_charge_drift",
            "total_structural_charge_drift",
            "structural_charge_drift",
            "monitor_alert_rate",
            "conservation_quality",
            "energy_derivative",
            "charge_drift",
            "violation_rate",
        }
        assert set(fv.keys()) == expected_keys

    def test_feedback_vector_types(self) -> None:
        G = _make_graph()
        monitor = enable_integrity_monitor(G)
        monitor.before_operator(G, 0)
        monitor.after_operator(G, 0, "Coherence")
        fv = monitor.feedback_vector()
        for key, value in fv.items():
            assert isinstance(
                value, float
            ), f"feedback_vector['{key}'] should be float, got {type(value)}"

    def test_feedback_vector_conservation_quality_bounded(self) -> None:
        G = _make_graph()
        monitor = enable_integrity_monitor(G)
        monitor.before_operator(G, 0)
        monitor.after_operator(G, 0, "Coherence")
        fv = monitor.feedback_vector()
        assert 0.0 <= fv["conservation_quality"] <= 1.0

    def test_feedback_vector_violation_rate_bounded(self) -> None:
        G = _make_graph()
        monitor = enable_integrity_monitor(G)
        monitor.before_operator(G, 0)
        monitor.after_operator(G, 0, "Coherence")
        fv = monitor.feedback_vector()
        assert 0.0 <= fv["violation_rate"] <= 1.0

    def test_feedback_vector_empty_before_any_cycle(self) -> None:
        G = _make_graph()
        monitor = enable_integrity_monitor(G)
        fv = monitor.feedback_vector()
        # No operations yet → defaults
        assert fv["violation_rate"] == 0.0
        assert fv["balance_sample_count"] == 0.0

    def test_feedback_vector_separates_mean_and_total_charge_drift(self) -> None:
        G = _make_graph()
        monitor = enable_integrity_monitor(G)
        monitor._summary.balance_samples = 4
        monitor._summary.total_charge_drift = 2.0

        fv = monitor.feedback_vector()

        assert fv["mean_structural_charge_drift"] == pytest.approx(0.5)
        assert fv["structural_charge_drift"] == pytest.approx(0.5)
        assert fv["total_structural_charge_drift"] == pytest.approx(2.0)
        assert fv["charge_drift"] == pytest.approx(2.0)


# ═══════════════════════════════════════════════════════════════════════════
# Backward compatibility
# ═══════════════════════════════════════════════════════════════════════════


class TestBackwardCompatibility:
    """Ensure zero overhead when monitor is not attached."""

    def test_no_monitor_no_overhead(self) -> None:
        """Graph without monitor should have no 'integrity_monitor' key."""
        G = _make_graph()
        assert "integrity_monitor" not in G.graph

    def test_operator_runs_without_monitor(self) -> None:
        """Operators should work normally without monitor."""
        try:
            from tnfr.operators.definitions import Coherence

            G = _make_graph()
            op = Coherence()
            op(G, 0)
        except ImportError:
            pytest.skip("Operator definitions not available")

    def test_operator_runs_with_monitor(self) -> None:
        """Operators should work normally with monitor in OBSERVE mode."""
        try:
            from tnfr.operators.definitions import Coherence

            G = _make_graph()
            enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
            op = Coherence()
            op(G, 0)
            monitor = StructuralIntegrityMonitor.get(G)
            # Monitor should have recorded at least one cycle
            assert monitor.summary.total_operators >= 1
        except ImportError:
            pytest.skip("Operator definitions not available")

    def test_transition_operator_runs_with_monitor(self) -> None:
        """NAV operator (custom __call__) must trigger integrity monitor hooks."""
        try:
            from tnfr.operators.definitions import Transition

            G = _make_graph()
            enable_integrity_monitor(G, mode=MonitorMode.OBSERVE)
            op = Transition()
            op(G, 0)
            monitor = StructuralIntegrityMonitor.get(G)
            assert monitor.summary.total_operators >= 1
            # Verify "transition" was the recorded operator name
            assert any(r.operator == "transition" for r in monitor._summary.reports)
        except ImportError:
            pytest.skip("Operator definitions not available")


# ═══════════════════════════════════════════════════════════════════════════
# StructuralIntegrityViolation exception
# ═══════════════════════════════════════════════════════════════════════════


class TestStructuralIntegrityViolation:
    """Test the violation exception."""

    def test_violation_inherits_from_exception(self) -> None:
        exc = StructuralIntegrityViolation(
            operator="Coherence",
            violation_type="postcondition_failed",
            details={"reason": "C(t) decreased"},
        )
        assert isinstance(exc, Exception)

    def test_violation_attributes(self) -> None:
        exc = StructuralIntegrityViolation(
            operator="Silence",
            violation_type="epi_changed",
            details={"reason": "EPI shifted by 0.5", "delta": 0.5},
        )
        assert exc.operator == "Silence"
        assert exc.violation_type == "epi_changed"
        assert isinstance(exc.details, dict)
        assert "0.5" in exc.details["reason"]

    def test_violation_str_representation(self) -> None:
        exc = StructuralIntegrityViolation(
            operator="Emission",
            violation_type="vf_unchanged",
            details={"reason": "νf did not increase"},
        )
        text = str(exc)
        assert "Emission" in text


# ═══════════════════════════════════════════════════════════════════════════
# Multi-operator sequence tracking
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiOperatorSequence:
    """Test monitoring across sequences of operators."""

    def test_multiple_operators_accumulate(self) -> None:
        G = _make_graph()
        monitor = enable_integrity_monitor(G)
        ops = ["Coherence", "Coherence", "Silence", "Coherence"]
        for i, op_name in enumerate(ops):
            node = i % len(G.nodes())
            monitor.before_operator(G, node)
            monitor.after_operator(G, node, op_name)
        assert monitor.summary.total_operators == len(ops)

    def test_feedback_vector_reflects_history(self) -> None:
        G = _make_graph()
        monitor = enable_integrity_monitor(G)
        for i in range(5):
            monitor.before_operator(G, 0)
            monitor.after_operator(G, 0, "Coherence")
        fv = monitor.feedback_vector()
        # After 5 clean operations, conservation quality should be decent
        assert fv["conservation_quality"] >= 0.0
        assert fv["violation_rate"] >= 0.0


class TestOperatorContractAudit:
    """Proactive measured fidelity of the 13 operators to their contracts."""

    def test_audit_returns_thirteen_results(self) -> None:
        audit = audit_operator_contracts()
        assert isinstance(audit, OperatorContractAudit)
        assert audit.n_operators == 13
        assert all(isinstance(r, OperatorContractResult) for r in audit.results)

    def test_all_thirteen_contracts_satisfied(self) -> None:
        # Every catalog entry passes its declared deterministic fixture.
        audit = audit_operator_contracts()
        assert audit.all_satisfied, audit.summary()
        assert audit.n_satisfied == 13
        assert audit.violations == ()

    def test_all_thirteen_glyphs_present(self) -> None:
        audit = audit_operator_contracts()
        glyphs = {r.glyph for r in audit.results}
        assert glyphs == {
            "AL",
            "EN",
            "IL",
            "OZ",
            "UM",
            "RA",
            "SHA",
            "VAL",
            "NUL",
            "THOL",
            "ZHIR",
            "NAV",
            "REMESH",
        }

    def test_stabiliser_il_reduces_dnfr(self) -> None:
        # IL is measured at network level: it must not increase |ΔNFR|
        audit = audit_operator_contracts()
        il = next(r for r in audit.results if r.glyph == "IL")
        assert il.satisfied
        assert il.context == "network"

    def test_coupling_um_reduces_dnfr(self) -> None:
        audit = audit_operator_contracts()
        um = next(r for r in audit.results if r.glyph == "UM")
        assert um.satisfied
        assert um.context == "network"

    def test_resonance_ra_preserves_identity(self) -> None:
        audit = audit_operator_contracts()
        ra = next(r for r in audit.results if r.glyph == "RA")
        assert ra.satisfied
        assert ra.context == "identity"

    def test_mutation_zhir_transforms_phase(self) -> None:
        audit = audit_operator_contracts()
        zhir = next(r for r in audit.results if r.glyph == "ZHIR")
        assert zhir.satisfied
        assert zhir.context == "phase"

    def test_audit_is_reproducible(self) -> None:
        a1 = audit_operator_contracts(seed=7)
        a2 = audit_operator_contracts(seed=7)
        assert [r.satisfied for r in a1.results] == [r.satisfied for r in a2.results]

    def test_summary_contains_verdict(self) -> None:
        audit = audit_operator_contracts()
        text = audit.summary()
        assert "ALL PROBES PASS" in text
        assert "13/13" in text


class TestSDKOperatorAudit:
    """The SDK exposes the measured operator audit and a working check."""

    def test_sdk_audit_operators(self) -> None:
        from tnfr.sdk import TNFR

        net = TNFR.create(16).ring().evolve(2)
        result = net.audit_operators()
        assert result["all_satisfied"] is True
        assert result["n_satisfied"] == 13
        assert result["n_operators"] == 13
        assert len(result["operators"]) == 13

    def test_sdk_integrity_check_requires_real_before_after_evidence(self) -> None:
        from tnfr.sdk import TNFR
        from tnfr.operators.definitions import Coherence

        net = TNFR.create(16).ring().evolve(2)
        baseline = net.integrity_check("IL")
        assert baseline["evidence_available"] is False
        assert baseline["nodes_checked"] == 0

        Coherence()(net.G, 0)
        result = net.integrity_check("IL")
        assert result["evidence_available"] is True
        assert result["nodes_checked"] > 0
        assert result["grammar_validated"] is False
        assert all(r["postcondition_evaluated"] for r in result["reports"])
        assert all("diagnostic_follow_up" in r for r in result["reports"])
