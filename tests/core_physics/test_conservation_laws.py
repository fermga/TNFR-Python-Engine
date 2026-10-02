"""Tests for declared structural-balance and energy diagnostics.

Finite snapshots supply rho = Phi_s + K_phi and currents (J_phi, J_DeltaNFR).
The tests check their implemented algebra, finite differences, bookkeeping and
scope metadata, including finite selected operator calls. Computed charge or
energy candidates need not be conserved by arbitrary nodal evolution. Residuals
and grammar labels do not derive U1-U6 or a general convergence theorem.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from tests.diagnostic_graph_fixtures import (
    make_diagnostic_field_graph as _make_tnfr_graph,
)
from tnfr.constants.canonical import PI
from tnfr.physics.conservation import (
    ConservationAlertLevels,
    ConservationBalance,
    ConservationSnapshot,
    ConservationTimeSeries,
    ConservationTracker,
    LyapunovResult,
    SpectralConservation,
    WardIdentity,
    analyze_sector_coupling,
    capture_conservation_snapshot,
    compute_charge_density,
    compute_current_divergence,
    compute_energy_functional,
    compute_grammar_conservation_bounds,
    compute_lyapunov_derivative,
    compute_noether_charge,
    compute_spectral_conservation,
    compute_ward_identity,
    decompose_conservation_residual,
    detect_grammar_violations_from_conservation,
    verify_conservation_balance,
    verify_sequence_ward_identity,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ws_graph():
    """Watts-Strogatz TNFR graph (30 nodes)."""
    return _make_tnfr_graph(30, "watts_strogatz")


@pytest.fixture
def ba_graph():
    """Barabási-Albert TNFR graph (30 nodes)."""
    return _make_tnfr_graph(30, "barabasi_albert")


@pytest.fixture
def grid_graph():
    """Grid TNFR graph (25 nodes, 5x5)."""
    return _make_tnfr_graph(25, "grid")


# ===========================================================================
# Test: Core data structures
# ===========================================================================


class TestConservationSnapshot:
    """Test snapshot capture produces valid data."""

    def test_snapshot_contains_all_fields(self, ws_graph):
        snap = capture_conservation_snapshot(ws_graph)
        nodes = list(ws_graph.nodes())

        assert len(snap.charge_density) == len(nodes)
        assert len(snap.phi_s) == len(nodes)
        assert len(snap.k_phi) == len(nodes)
        assert len(snap.j_phi) == len(nodes)
        assert len(snap.j_dnfr) == len(nodes)
        assert len(snap.grad_phi) == len(nodes)
        assert len(snap.divergence) == len(nodes)

    def test_charge_density_equals_phi_s_plus_k_phi(self, ws_graph):
        snap = capture_conservation_snapshot(ws_graph)
        for n in ws_graph.nodes():
            expected = snap.phi_s[n] + snap.k_phi[n]
            assert abs(snap.charge_density[n] - expected) < 1e-12

    def test_snapshot_is_frozen(self, ws_graph):
        snap = capture_conservation_snapshot(ws_graph)
        with pytest.raises(AttributeError):
            snap.charge_density = {}  # type: ignore[misc]


# ===========================================================================
# Test: Charge density computation
# ===========================================================================


class TestChargeDensity:
    """Test ρ(i) = Φ_s(i) + K_φ(i)."""

    @pytest.mark.parametrize("topo", ["watts_strogatz", "barabasi_albert", "grid"])
    def test_charge_density_across_topologies(self, topo):
        G = _make_tnfr_graph(25, topo)
        rho = compute_charge_density(G)
        assert len(rho) == G.number_of_nodes()
        assert all(math.isfinite(v) for v in rho.values())


# ===========================================================================
# Test: Current divergence
# ===========================================================================


class TestCurrentDivergence:
    """Test div J(i) = div(J_φ, J_ΔNFR)."""

    def test_neighbor_mean_divergence_has_degree_weighted_not_uniform_balance(self):
        """The irregular P3 current is (2,-2.5,3), so div=(-4.5,5,-5.5)."""
        graph = nx.path_graph(3)
        nx.set_node_attributes(graph, 0.0, "phase")
        nx.set_node_attributes(graph, {0: 0.0, 1: 2.0, 2: -1.0}, "delta_nfr")
        divergence = compute_current_divergence(graph)
        assert divergence == {0: -4.5, 1: 5.0, 2: -5.5}
        assert sum(divergence.values()) == -5.0
        assert sum(graph.degree[node] * divergence[node] for node in graph) == 0.0


# ===========================================================================
# Test: Conservation balance (static)
# ===========================================================================


class TestConservationBalance:
    """Test the two-snapshot balance verification."""

    def test_identical_snapshots_retain_static_divergence(self, ws_graph):
        """Same state → zero ∂ρ/∂t → residual = div J only."""
        snap = capture_conservation_snapshot(ws_graph)
        balance = verify_conservation_balance(snap, snap)

        # ∂ρ/∂t = 0 for identical snapshots
        for n in ws_graph.nodes():
            assert abs(balance.delta_rho[n]) < 1e-12

        # Charge drift must be zero
        assert balance.charge_drift < 1e-12
        assert balance.residual == snap.divergence
        expected_rms = math.sqrt(np.mean(np.square(list(snap.divergence.values()))))
        assert balance.conservation_quality == pytest.approx(1.0 / (1.0 + expected_rms))

    def test_balance_has_correct_total_charges(self, ws_graph):
        snap = capture_conservation_snapshot(ws_graph)
        balance = verify_conservation_balance(snap, snap)
        expected_q = sum(snap.charge_density.values())
        assert abs(balance.total_charge_before - expected_q) < 1e-12
        assert abs(balance.total_charge_after - expected_q) < 1e-12

    def test_balance_exposes_non_grammar_scope(self, ws_graph):
        snap = capture_conservation_snapshot(ws_graph)
        balance = verify_conservation_balance(snap, snap)
        assert balance.diagnostic_scope == "two_snapshot_structural_balance"
        assert balance.grammar_validation_applicable is False
        assert balance.assessed_grammar_rules == ()
        assert balance.balance_alert_index == balance.grammar_violation_index


# ===========================================================================
# Test: Noether charge
# ===========================================================================


class TestNoetherCharge:
    """Test Q = Σ_i ρ(i) is well-defined and finite."""

    def test_noether_charge_equals_sum_of_charge_density(self, ws_graph):
        Q = compute_noether_charge(ws_graph)
        rho = compute_charge_density(ws_graph)
        assert abs(Q - sum(rho.values())) < 1e-12


# ===========================================================================
# Test: Energy functional
# ===========================================================================


class TestEnergyFunctional:
    """Test E = (1/2)Σ(Φ_s² + K_φ² + J_φ² + J_ΔNFR²) ≥ 0."""

    def test_energy_matches_independent_pressure_only_path(self):
        graph = nx.path_graph(3)
        nx.set_node_attributes(graph, 0.0, "phase")
        nx.set_node_attributes(graph, {0: 0.0, 1: 2.0, 2: -1.0}, "delta_nfr")
        # Potential=(1.75,-1,2), pressure current=(2,-2.5,3); phase fields vanish.
        expected = (1.75**2 + 1.0 + 4.0 + 4.0 + 2.5**2 + 9.0) / 2.0
        assert compute_energy_functional(graph) == expected


# ===========================================================================
# Test: Grammar bounds
# ===========================================================================


class TestGrammarBounds:
    """Test backward-compatible legacy pi-scaled alert levels."""

    def test_bounds_contain_phi_confinement(self, ws_graph):
        bounds = compute_grammar_conservation_bounds(ws_graph)
        # Legacy key retained; the value is the U6 drift alert scale.
        assert bounds["phi_s_confinement"] == pytest.approx(PI / 2, rel=1e-10)

    def test_alert_metadata_rejects_bound_and_grammar_semantics(self, ws_graph):
        levels = compute_grammar_conservation_bounds(ws_graph)
        assert isinstance(levels, ConservationAlertLevels)
        assert isinstance(levels, dict)
        assert levels.metadata["thresholds_are_proven_bounds"] is False
        assert levels.metadata["grammar_validation_applicable"] is False
        assert levels.metadata["applicable_grammar_rules"] == ()
        assert levels.metadata["u6_drift_requires_reference"] is True

    def test_bounds_are_positive(self, ws_graph):
        bounds = compute_grammar_conservation_bounds(ws_graph)
        for key, val in bounds.items():
            assert val > 0, f"Bound {key} should be positive, got {val}"

    def test_max_charge_equals_phi_plus_pi(self, ws_graph):
        bounds = compute_grammar_conservation_bounds(ws_graph)
        # Historical composite alert retains its numerical API.
        expected = PI / 2 + PI
        assert bounds["max_charge_density"] == pytest.approx(expected, rel=1e-10)


# ===========================================================================
# Test: Conservation tracker (multi-step)
# ===========================================================================


class TestConservationTracker:
    """Test the tracker across multiple steps."""

    def test_tracker_initial_record(self, ws_graph):
        tracker = ConservationTracker(ws_graph)
        snap = tracker.record(t=0.0)
        assert isinstance(snap, ConservationSnapshot)
        assert tracker.latest_balance is None  # Only one snapshot
        report = tracker.report()
        assert report.mean_quality == 0.0
        assert report.sampled_mean_quality == 0.0
        assert report.mean_quality_including_baseline == 1.0
        assert not report.aggregate_balance_within_alert

    def test_tracker_two_records(self, ws_graph):
        tracker = ConservationTracker(ws_graph)
        tracker.record(t=0.0)

        # Small perturbation
        for n in ws_graph.nodes():
            ws_graph.nodes[n]["delta_nfr"] *= 1.001

        tracker.record(t=1.0)
        balance = tracker.latest_balance
        assert balance is not None
        assert isinstance(balance, ConservationBalance)

    def test_tracker_report(self, ws_graph):
        tracker = ConservationTracker(ws_graph)
        rng = np.random.default_rng(123)

        for step in range(5):
            tracker.record(t=float(step))
            # Small random walk in ΔNFR
            for n in ws_graph.nodes():
                ws_graph.nodes[n]["delta_nfr"] += rng.uniform(-0.01, 0.01)

        report = tracker.report()
        assert isinstance(report, ConservationTimeSeries)
        assert len(report.times) == 5
        assert len(report.total_charge) == 5
        assert isinstance(report.mean_quality, float)
        assert report.mean_quality == pytest.approx(
            np.mean(report.conservation_quality[1:])
        )
        assert report.mean_quality_including_baseline == pytest.approx(
            np.mean(report.conservation_quality)
        )


# ===========================================================================
# Test: Sector decomposition
# ===========================================================================


class TestSectorCoupling:
    """Test decomposition into potential and geometric sectors."""

    def test_decomposition_keys(self, ws_graph):
        before = capture_conservation_snapshot(ws_graph)
        for n in ws_graph.nodes():
            ws_graph.nodes[n]["delta_nfr"] *= 1.01
        after = capture_conservation_snapshot(ws_graph)

        decomp = decompose_conservation_residual(before, after)
        expected_keys = {
            "phi_s_drift",
            "k_phi_drift",
            "j_phi_div",
            "j_dnfr_div",
            "potential_residual",
            "geometric_residual",
        }
        assert set(decomp.keys()) == expected_keys

    def test_sector_analysis_returns_valid_data(self, ws_graph):
        before = capture_conservation_snapshot(ws_graph)
        for n in ws_graph.nodes():
            ws_graph.nodes[n]["delta_nfr"] *= 1.01
        after = capture_conservation_snapshot(ws_graph)

        result = analyze_sector_coupling(before, after)
        assert result["dominant_sector"] in ("potential", "geometric", "balanced")
        assert result["sector_asymmetry"] >= 1.0
        assert -1.0 <= result["cross_coupling_strength"] <= 1.0


# ===========================================================================
# Test: Grammar violation detection
# ===========================================================================


class TestGrammarViolationDetection:
    """Test that the legacy entry point reports alerts, not grammar verdicts."""

    def test_no_violations_on_static_state(self, ws_graph):
        snap = capture_conservation_snapshot(ws_graph)
        balance = verify_conservation_balance(snap, snap)
        result = detect_grammar_violations_from_conservation(balance)
        assert not result["violations_detected"]

    def test_extreme_perturbation_triggers_violation(self, ws_graph):
        before = capture_conservation_snapshot(ws_graph)
        for n in ws_graph.nodes():
            ws_graph.nodes[n]["delta_nfr"] = 100.0  # extreme
        after = capture_conservation_snapshot(ws_graph)

        balance = verify_conservation_balance(before, after)
        bounds = compute_grammar_conservation_bounds(ws_graph)
        result = detect_grammar_violations_from_conservation(balance, bounds)
        # Extreme perturbation produces a balance alert, never an inferred
        # U2/U3/U6 classification.
        assert result["severity"] > 0.0
        assert result["alerts_detected"]
        assert result["alert_count"] > 0
        assert not result["violations_detected"]
        assert result["violation_types"] == []
        assert result["nodes_violating"] == []
        assert result["grammar_validation_applicable"] is False
        assert result["grammar_validated"] is False
        assert result["grammar_rules_assessed"] == ()
        assert result["thresholds_are_proven_bounds"] is False
        assert all(not label.startswith("U") for label in result["alert_types"])


# ===========================================================================
# Test: Cross-topology consistency
# ===========================================================================


class TestCrossTopologyConsistency:
    """Verify conservation law works across different topologies."""

    @pytest.mark.parametrize("topo", ["watts_strogatz", "barabasi_albert", "grid"])
    def test_conservation_snapshot_across_topologies(self, topo):
        G = _make_tnfr_graph(25, topo)
        snap = capture_conservation_snapshot(G)
        assert len(snap.charge_density) == G.number_of_nodes()

    @pytest.mark.parametrize("topo", ["watts_strogatz", "barabasi_albert", "grid"])
    def test_noether_charge_consistent(self, topo):
        G = _make_tnfr_graph(25, topo)
        Q = compute_noether_charge(G)
        rho = compute_charge_density(G)
        assert abs(Q - sum(rho.values())) < 1e-12

    @pytest.mark.parametrize("topo", ["watts_strogatz", "barabasi_albert", "grid"])
    def test_energy_positive_all_topologies(self, topo):
        G = _make_tnfr_graph(25, topo)
        E = compute_energy_functional(G)
        assert E > 0.0


# ===========================================================================
# Test: Ward identities
# ===========================================================================


class TestWardIdentity:
    """Test per-operator conservation signatures."""

    def test_ward_identity_for_static_step(self, ws_graph):
        """Identical endpoints have no charge/energy change, but can have divergence."""
        snap = capture_conservation_snapshot(ws_graph)
        ward = compute_ward_identity(snap, snap, operator_name="SHA")
        assert isinstance(ward, WardIdentity)
        assert ward.operator_name == "SHA"
        assert abs(ward.delta_charge) < 1e-12
        assert ward.charge_character == "exact"
        assert ward.energy_character == "neutral"
        assert ward.delta_energy == 0.0
        assert ward.mean_source == pytest.approx(
            np.mean(list(snap.divergence.values()))
        )

    def test_ward_mean_source_uses_rate_and_trapezoidal_divergence(self, ws_graph):
        before = capture_conservation_snapshot(ws_graph)
        for node in ws_graph.nodes():
            ws_graph.nodes[node]["delta_nfr"] += 0.2
        after = capture_conservation_snapshot(ws_graph)
        dt = 0.25
        ward = compute_ward_identity(before, after, "observed_step", dt=dt)

        expected = ward.delta_charge / (len(ws_graph) * dt) + np.mean(
            [
                0.5 * (before.divergence[node] + after.divergence[node])
                for node in ws_graph
            ]
        )
        assert ward.mean_source == pytest.approx(expected)

    @pytest.mark.parametrize("threshold", [0.0, -1.0, math.nan, math.inf])
    def test_ward_rejects_invalid_classification_threshold(self, ws_graph, threshold):
        snap = capture_conservation_snapshot(ws_graph)
        with pytest.raises(ValueError, match="threshold"):
            compute_ward_identity(snap, snap, "SHA", threshold=threshold)

    def test_sequence_ward_identity(self, ws_graph):
        """Sequence of small perturbations should approximately conserve."""
        identities = []
        rng = np.random.default_rng(42)
        for step in range(5):
            before = capture_conservation_snapshot(ws_graph)
            for n in ws_graph.nodes():
                ws_graph.nodes[n]["delta_nfr"] += rng.uniform(-0.01, 0.01)
            after = capture_conservation_snapshot(ws_graph)
            ward = compute_ward_identity(before, after, operator_name=f"step_{step}")
            identities.append(ward)

        result = verify_sequence_ward_identity(identities)
        assert "total_source" in result
        assert "total_charge_change" in result
        assert "total_energy_change" in result
        assert "sequence_conserved" in result
        assert isinstance(result["operator_summary"], dict)
        assert result["sequence_conserved"] == result["aggregate_balance_within_alert"]
        assert result["thresholds_are_proven_bounds"] is False
        assert result["grammar_validation_applicable"] is False
        assert result["grammar_validated"] is False

    def test_sequence_ward_accepts_explicit_alert_level(self, ws_graph):
        snap = capture_conservation_snapshot(ws_graph)
        ward = compute_ward_identity(snap, snap, operator_name="observed_step")
        result = verify_sequence_ward_identity([ward], alert_level=0.25)
        assert result["alert_threshold"] == pytest.approx(0.25)

    @pytest.mark.parametrize("alert_level", [0.0, -1.0, math.nan, math.inf])
    def test_sequence_ward_rejects_invalid_alert_level(self, alert_level):
        with pytest.raises(ValueError, match="alert_level"):
            verify_sequence_ward_identity([], alert_level=alert_level)


# ===========================================================================
# Test: Lyapunov stability
# ===========================================================================


class TestLyapunovDerivative:
    """Test Lyapunov derivative dE/dt computation."""

    def test_lyapunov_static_is_stable(self, ws_graph):
        """Identical snapshots => dE/dt = 0 => stable."""
        snap = capture_conservation_snapshot(ws_graph)
        lyap = compute_lyapunov_derivative(snap, snap)
        assert isinstance(lyap, LyapunovResult)
        assert abs(lyap.energy_derivative) < 1e-12
        assert lyap.is_stable

    def test_lyapunov_energy_before_after(self, ws_graph):
        """Energy values should be non-negative and finite."""
        before = capture_conservation_snapshot(ws_graph)
        rng = np.random.default_rng(42)
        for n in ws_graph.nodes():
            ws_graph.nodes[n]["delta_nfr"] += rng.uniform(-0.01, 0.01)
        after = capture_conservation_snapshot(ws_graph)

        lyap = compute_lyapunov_derivative(before, after)
        assert lyap.energy_before >= 0.0
        assert lyap.energy_after >= 0.0
        assert math.isfinite(lyap.energy_derivative)


# ===========================================================================
# Test: Spectral conservation
# ===========================================================================


class TestSpectralConservation:
    """Test spectral decomposition of conservation fields."""

    def test_spectral_returns_valid_data(self, ws_graph):
        spec = compute_spectral_conservation(ws_graph)
        assert isinstance(spec, SpectralConservation)
        n = ws_graph.number_of_nodes()
        assert len(spec.eigenvalues) == n
        assert len(spec.rho_spectrum) == n
        assert len(spec.div_spectrum) == n
        assert len(spec.conservation_by_mode) == n

    def test_zero_mode_has_zero_eigenvalue(self, ws_graph):
        """First eigenvalue of graph Laplacian is 0 (connected graph)."""
        spec = compute_spectral_conservation(ws_graph)
        assert abs(spec.eigenvalues[0]) < 1e-10

    def test_conservation_modes_count(self, ws_graph):
        spec = compute_spectral_conservation(ws_graph)
        n = ws_graph.number_of_nodes()
        assert 0 < spec.dominant_conservation_modes <= n


# ===========================================================================
# Test: Operator-event observation integration
# ===========================================================================


class TestOperatorBalanceIntegration:
    """Check finite diagnostics for sequences carrying U2 role labels.

    The labels organize grammar debt; they do not prove convergence of the
    nodal integral or fix the sign of the structural-energy candidate.
    """

    @staticmethod
    def _make_graph(
        n: int = 30, topo: str = "watts_strogatz", seed: int = 42
    ) -> nx.Graph:
        G = _make_tnfr_graph(n, topo, seed=seed)
        # Numeric EPI required for operator invocation (BEPI element)
        for nd in G.nodes():
            G.nodes[nd]["EPI"] = 1.0
        return G

    def test_operator_events_are_recorded_in_tracker(self):
        """Track actual AL/OZ/IL/SHA events without imposing energy-sign claims."""
        from tnfr.operators.definitions import Coherence, Dissonance, Emission, Silence

        G = self._make_graph()
        node = list(G.nodes())[0]
        tracker = ConservationTracker(G)
        tracker.record(t=0.0)

        Emission()(G, node)
        tracker.record(t=1.0)

        Dissonance()(G, node)
        tracker.record(t=2.0)

        Coherence()(G, node)
        tracker.record(t=3.0)

        Silence()(G, node)
        tracker.record(t=4.0)

        report = tracker.report()
        # All steps should produce finite diagnostics
        assert all(math.isfinite(q) for q in report.total_charge)
        assert all(math.isfinite(r) for r in report.rms_residuals)
        assert all(0 <= q <= 1.0 for q in report.conservation_quality)
        assert report.times == [0.0, 1.0, 2.0, 3.0, 4.0]
        assert report.total_charge[-1] == pytest.approx(compute_noether_charge(G))
        assert tracker.latest_balance.charge_drift == pytest.approx(
            abs(report.total_charge[-1] - report.total_charge[-2])
        )
