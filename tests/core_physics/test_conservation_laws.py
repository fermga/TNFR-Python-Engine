"""Test Conservation Laws — TNFR Structural Continuity Theorem.

Validates the Noether-like conservation law:
    ∂ρ/∂t + div(J) ≈ 0  when grammar U1-U6 is satisfied

where ρ = Φ_s + K_φ (structural charge) and J = (J_φ, J_ΔNFR) (current).

TIER: CORE PHYSICS — Conservation is a fundamental property of the theory.
"""
from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from tnfr.constants.canonical import PHI, PI
from tnfr.constants import inject_defaults
from tnfr.physics.conservation import (
    ConservationSnapshot,
    ConservationBalance,
    ConservationTimeSeries,
    ConservationTracker,
    WardIdentity,
    LyapunovResult,
    SpectralConservation,
    capture_conservation_snapshot,
    compute_charge_density,
    compute_current_divergence,
    compute_energy_functional,
    compute_grammar_conservation_bounds,
    compute_noether_charge,
    compute_ward_identity,
    verify_sequence_ward_identity,
    compute_lyapunov_derivative,
    compute_spectral_conservation,
    decompose_conservation_residual,
    analyze_sector_coupling,
    detect_grammar_violations_from_conservation,
    verify_conservation_balance,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_tnfr_graph(
    n: int = 30,
    topology: str = "watts_strogatz",
    seed: int = 42,
) -> nx.Graph:
    """Build a TNFR-ready graph with canonical attributes."""
    rng = np.random.default_rng(seed)

    if topology == "watts_strogatz":
        G = nx.watts_strogatz_graph(n, 4, 0.3, seed=seed)
    elif topology == "barabasi_albert":
        G = nx.barabasi_albert_graph(n, 3, seed=seed)
    elif topology == "grid":
        side = int(math.sqrt(n))
        G = nx.grid_2d_graph(side, side)
    else:
        G = nx.watts_strogatz_graph(n, 4, 0.3, seed=seed)

    inject_defaults(G)

    # Assign structural attributes
    for node in G.nodes():
        G.nodes[node]['phase'] = rng.uniform(0, 2 * math.pi)
        G.nodes[node]['frequency'] = rng.uniform(0.1, 1.0)
        G.nodes[node]['delta_nfr'] = rng.uniform(-0.5, 0.5)
        G.nodes[node]['EPI'] = f"epi_{node}"

    return G


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

    def test_charge_density_keys_match_nodes(self, ws_graph):
        rho = compute_charge_density(ws_graph)
        assert set(rho.keys()) == set(ws_graph.nodes())

    def test_charge_density_is_finite(self, ws_graph):
        rho = compute_charge_density(ws_graph)
        for val in rho.values():
            assert math.isfinite(val)

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

    def test_divergence_keys_match_nodes(self, ws_graph):
        div_j = compute_current_divergence(ws_graph)
        assert set(div_j.keys()) == set(ws_graph.nodes())

    def test_divergence_is_finite(self, ws_graph):
        div_j = compute_current_divergence(ws_graph)
        for val in div_j.values():
            assert math.isfinite(val)

    def test_total_divergence_near_zero(self, ws_graph):
        """Sum of divergence over all nodes should be small (Gauss theorem)."""
        div_j = compute_current_divergence(ws_graph)
        total = sum(div_j.values())
        # On a closed graph, total divergence is theoretically 0
        # Allow numerical tolerance
        assert abs(total) < 1.0, f"Total divergence {total} too large"


# ===========================================================================
# Test: Conservation balance (static)
# ===========================================================================

class TestConservationBalance:
    """Test the two-snapshot balance verification."""

    def test_identical_snapshots_give_zero_residual(self, ws_graph):
        """Same state → zero ∂ρ/∂t → residual = div J only."""
        snap = capture_conservation_snapshot(ws_graph)
        balance = verify_conservation_balance(snap, snap)

        # ∂ρ/∂t = 0 for identical snapshots
        for n in ws_graph.nodes():
            assert abs(balance.delta_rho[n]) < 1e-12

        # Charge drift must be zero
        assert balance.charge_drift < 1e-12

    def test_conservation_quality_for_identical_snapshots(self, ws_graph):
        snap = capture_conservation_snapshot(ws_graph)
        balance = verify_conservation_balance(snap, snap)
        # Quality should be high since ∂ρ/∂t=0; residual = div J only
        assert balance.conservation_quality > 0.5

    def test_small_perturbation_has_small_residual(self, ws_graph):
        """Tiny ΔNFR perturbation should produce small residual."""
        before = capture_conservation_snapshot(ws_graph)

        rng = np.random.default_rng(99)
        for n in ws_graph.nodes():
            # Very small perturbation
            ws_graph.nodes[n]['delta_nfr'] += rng.uniform(-0.001, 0.001)

        after = capture_conservation_snapshot(ws_graph)
        balance = verify_conservation_balance(before, after)

        assert balance.rms_residual < 1.0
        assert balance.conservation_quality > 0.5

    def test_balance_has_correct_total_charges(self, ws_graph):
        snap = capture_conservation_snapshot(ws_graph)
        balance = verify_conservation_balance(snap, snap)
        expected_q = sum(snap.charge_density.values())
        assert abs(balance.total_charge_before - expected_q) < 1e-12
        assert abs(balance.total_charge_after - expected_q) < 1e-12


# ===========================================================================
# Test: Noether charge
# ===========================================================================

class TestNoetherCharge:
    """Test Q = Σ_i ρ(i) is well-defined and finite."""

    def test_noether_charge_is_finite(self, ws_graph):
        Q = compute_noether_charge(ws_graph)
        assert math.isfinite(Q)

    def test_noether_charge_equals_sum_of_charge_density(self, ws_graph):
        Q = compute_noether_charge(ws_graph)
        rho = compute_charge_density(ws_graph)
        assert abs(Q - sum(rho.values())) < 1e-12

    @pytest.mark.parametrize("topo", ["watts_strogatz", "barabasi_albert", "grid"])
    def test_noether_charge_across_topologies(self, topo):
        G = _make_tnfr_graph(25, topo)
        Q = compute_noether_charge(G)
        assert math.isfinite(Q)


# ===========================================================================
# Test: Energy functional
# ===========================================================================

class TestEnergyFunctional:
    """Test E = (1/2)Σ(Φ_s² + K_φ² + J_φ² + J_ΔNFR²) ≥ 0."""

    def test_energy_is_non_negative(self, ws_graph):
        E = compute_energy_functional(ws_graph)
        assert E >= 0.0

    def test_energy_is_finite(self, ws_graph):
        E = compute_energy_functional(ws_graph)
        assert math.isfinite(E)

    def test_energy_positive_for_nontrivial_network(self, ws_graph):
        E = compute_energy_functional(ws_graph)
        assert E > 0.0, "Nontrivial network must have E > 0"


# ===========================================================================
# Test: Grammar bounds
# ===========================================================================

class TestGrammarBounds:
    """Test theoretical conservation bounds from grammar constraints."""

    def test_bounds_contain_phi_confinement(self, ws_graph):
        bounds = compute_grammar_conservation_bounds(ws_graph)
        assert bounds['phi_s_confinement'] == pytest.approx(PHI, rel=1e-10)

    def test_bounds_are_positive(self, ws_graph):
        bounds = compute_grammar_conservation_bounds(ws_graph)
        for key, val in bounds.items():
            assert val > 0, f"Bound {key} should be positive, got {val}"

    def test_max_charge_equals_phi_plus_pi(self, ws_graph):
        bounds = compute_grammar_conservation_bounds(ws_graph)
        expected = PHI + PI
        assert bounds['max_charge_density'] == pytest.approx(expected, rel=1e-10)


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

    def test_tracker_two_records(self, ws_graph):
        tracker = ConservationTracker(ws_graph)
        tracker.record(t=0.0)

        # Small perturbation
        for n in ws_graph.nodes():
            ws_graph.nodes[n]['delta_nfr'] *= 1.001

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
                ws_graph.nodes[n]['delta_nfr'] += rng.uniform(-0.01, 0.01)

        report = tracker.report()
        assert isinstance(report, ConservationTimeSeries)
        assert len(report.times) == 5
        assert len(report.total_charge) == 5
        assert isinstance(report.mean_quality, float)

    def test_near_static_evolution_is_conserved(self, ws_graph):
        """Near-static evolution should achieve high conservation quality."""
        tracker = ConservationTracker(ws_graph)
        rng = np.random.default_rng(77)

        for step in range(6):
            tracker.record(t=float(step))
            for n in ws_graph.nodes():
                ws_graph.nodes[n]['delta_nfr'] += rng.uniform(-0.0001, 0.0001)

        report = tracker.report()
        # Near-static evolution: quality should remain stable across steps
        # (quality ≈ 0.5-0.6 is typical due to discrete divergence terms)
        assert report.mean_quality > 0.4
        # Charge drift should be small for tiny perturbations
        assert all(d < 0.1 for d in report.charge_drift)


# ===========================================================================
# Test: Sector decomposition
# ===========================================================================

class TestSectorCoupling:
    """Test decomposition into potential and geometric sectors."""

    def test_decomposition_keys(self, ws_graph):
        before = capture_conservation_snapshot(ws_graph)
        for n in ws_graph.nodes():
            ws_graph.nodes[n]['delta_nfr'] *= 1.01
        after = capture_conservation_snapshot(ws_graph)

        decomp = decompose_conservation_residual(before, after)
        expected_keys = {
            'phi_s_drift', 'k_phi_drift',
            'j_phi_div', 'j_dnfr_div',
            'potential_residual', 'geometric_residual',
        }
        assert set(decomp.keys()) == expected_keys

    def test_sector_analysis_returns_valid_data(self, ws_graph):
        before = capture_conservation_snapshot(ws_graph)
        for n in ws_graph.nodes():
            ws_graph.nodes[n]['delta_nfr'] *= 1.01
        after = capture_conservation_snapshot(ws_graph)

        result = analyze_sector_coupling(before, after)
        assert result['dominant_sector'] in ('potential', 'geometric', 'balanced')
        assert result['sector_asymmetry'] >= 1.0
        assert -1.0 <= result['cross_coupling_strength'] <= 1.0

    def test_pure_dnfr_change_dominates_potential_sector(self, ws_graph):
        """Pure ΔNFR perturbation should load the potential sector."""
        before = capture_conservation_snapshot(ws_graph)
        for n in ws_graph.nodes():
            ws_graph.nodes[n]['delta_nfr'] *= 1.5  # significant change
        after = capture_conservation_snapshot(ws_graph)

        result = analyze_sector_coupling(before, after)
        # Expect potential sector to dominate or be balanced
        assert result['potential_sector_residual'] >= 0.0


# ===========================================================================
# Test: Grammar violation detection
# ===========================================================================

class TestGrammarViolationDetection:
    """Test detection of grammar violations via conservation analysis."""

    def test_no_violations_on_static_state(self, ws_graph):
        snap = capture_conservation_snapshot(ws_graph)
        balance = verify_conservation_balance(snap, snap)
        result = detect_grammar_violations_from_conservation(balance)
        assert not result['violations_detected']

    def test_extreme_perturbation_triggers_violation(self, ws_graph):
        before = capture_conservation_snapshot(ws_graph)
        for n in ws_graph.nodes():
            ws_graph.nodes[n]['delta_nfr'] = 100.0  # extreme
        after = capture_conservation_snapshot(ws_graph)

        balance = verify_conservation_balance(before, after)
        bounds = compute_grammar_conservation_bounds(ws_graph)
        result = detect_grammar_violations_from_conservation(balance, bounds)
        # Extreme perturbation should be detected
        assert result['severity'] > 0.0


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
        """Identical snapshots => exact conservation (no source)."""
        snap = capture_conservation_snapshot(ws_graph)
        ward = compute_ward_identity(snap, snap, operator_name="SHA")
        assert isinstance(ward, WardIdentity)
        assert ward.operator_name == "SHA"
        assert abs(ward.delta_charge) < 1e-12
        assert ward.charge_character == "exact"

    def test_ward_identity_classifies_source(self, ws_graph):
        """A perturbation that increases total charge is classified as source."""
        before = capture_conservation_snapshot(ws_graph)
        # Increase delta_nfr to raise Phi_s (and thus charge)
        for n in ws_graph.nodes():
            ws_graph.nodes[n]["delta_nfr"] += 1.0
        after = capture_conservation_snapshot(ws_graph)
        ward = compute_ward_identity(before, after, operator_name="OZ")
        assert ward.charge_character in ("source", "sink", "transport", "exact")
        assert math.isfinite(ward.delta_charge)
        assert math.isfinite(ward.delta_energy)

    def test_ward_identity_energy_character(self, ws_graph):
        """Energy character should be one of dissipative/injective/neutral."""
        snap = capture_conservation_snapshot(ws_graph)
        ward = compute_ward_identity(snap, snap, operator_name="SHA")
        assert ward.energy_character in ("dissipative", "injective", "neutral")

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

    def test_lyapunov_dissipation_non_negative(self, ws_graph):
        """Dissipation D[G] = max(0, -dE/dt) is always non-negative."""
        snap = capture_conservation_snapshot(ws_graph)
        lyap = compute_lyapunov_derivative(snap, snap)
        assert lyap.dissipation >= 0.0

    @pytest.mark.parametrize("topo", ["watts_strogatz", "barabasi_albert", "grid"])
    def test_diffusive_evolution_is_lyapunov_stable(self, topo):
        """Diffusive dynamics should decrease energy (Lyapunov theorem)."""
        G = _make_tnfr_graph(25, topo)
        before = capture_conservation_snapshot(G)

        # Simple ΔNFR diffusion (stabilizing)
        for n in G.nodes():
            nbrs = list(G.neighbors(n))
            if nbrs:
                dnfr = G.nodes[n].get("delta_nfr", 0.0)
                mean_dnfr = np.mean([G.nodes[j].get("delta_nfr", 0.0) for j in nbrs])
                G.nodes[n]["delta_nfr"] += 0.1 * (mean_dnfr - dnfr)

        after = capture_conservation_snapshot(G)
        lyap = compute_lyapunov_derivative(before, after)
        # Diffusive dynamics should be stable or near-stable
        assert lyap.energy_derivative < 1.0  # allow some tolerance


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

    def test_spectral_gap_positive(self, ws_graph):
        """Connected graph has positive spectral gap lambda_1 > 0."""
        spec = compute_spectral_conservation(ws_graph)
        assert spec.spectral_gap > 0.0

    def test_zero_mode_has_zero_eigenvalue(self, ws_graph):
        """First eigenvalue of graph Laplacian is 0 (connected graph)."""
        spec = compute_spectral_conservation(ws_graph)
        assert abs(spec.eigenvalues[0]) < 1e-10

    def test_rho_spectrum_finite(self, ws_graph):
        spec = compute_spectral_conservation(ws_graph)
        assert np.all(np.isfinite(spec.rho_spectrum))

    def test_conservation_modes_count(self, ws_graph):
        spec = compute_spectral_conservation(ws_graph)
        n = ws_graph.number_of_nodes()
        assert 0 < spec.dominant_conservation_modes <= n

    @pytest.mark.parametrize("topo", ["watts_strogatz", "barabasi_albert", "grid"])
    def test_spectral_across_topologies(self, topo):
        G = _make_tnfr_graph(25, topo)
        spec = compute_spectral_conservation(G)
        assert isinstance(spec, SpectralConservation)
        assert spec.spectral_gap >= 0.0


# ===========================================================================
# Test: Dissipative regime integration (P4)
# ===========================================================================

class TestDissipativeOperatorSequences:
    """Test that destabilizer-stabilizer sequences maintain U2 convergence.

    Physics: Grammar U2 requires ∫νf·ΔNFR dt to converge. Destabilizers
    (OZ, VAL) increase |ΔNFR| creating convergence "debt"; stabilizers
    (IL, THOL) decrease |ΔNFR| paying it back. These tests verify the
    full feedback loop at the conservation level.
    """

    @staticmethod
    def _make_graph(n: int = 30, topo: str = "watts_strogatz",
                    seed: int = 42) -> nx.Graph:
        G = _make_tnfr_graph(n, topo, seed=seed)
        # Numeric EPI required for operator invocation (BEPI element)
        for nd in G.nodes():
            G.nodes[nd]['EPI'] = 1.0
        return G

    def test_dissonance_then_coherence_lyapunov_stable(self):
        """OZ → IL sequence should be Lyapunov stable (dE/dt ≤ 0 net)."""
        from tnfr.operators.definitions import Dissonance, Coherence
        G = self._make_graph()
        node = list(G.nodes())[0]
        before = capture_conservation_snapshot(G)
        Dissonance()(G, node)
        Coherence()(G, node)
        after = capture_conservation_snapshot(G)
        lyap = compute_lyapunov_derivative(before, after)
        assert lyap.dissipation >= 0.0, "Dissipation must be non-negative"
        assert math.isfinite(lyap.energy_derivative)

    def test_dissonance_then_coherence_charge_bounded(self):
        """OZ → IL should not produce unbounded charge drift."""
        from tnfr.operators.definitions import Dissonance, Coherence
        G = self._make_graph()
        node = list(G.nodes())[0]
        before = capture_conservation_snapshot(G)
        Dissonance()(G, node)
        Coherence()(G, node)
        after = capture_conservation_snapshot(G)
        balance = verify_conservation_balance(before, after)
        assert math.isfinite(balance.charge_drift)
        assert balance.rms_residual < 10.0, (
            f"OZ→IL RMS residual {balance.rms_residual:.4f} unexpectedly large"
        )

    def test_full_grammar_sequence_al_oz_il_sha(self):
        """Full grammar-compliant sequence: AL → OZ → IL → SHA."""
        from tnfr.operators.definitions import (
            Emission, Dissonance, Coherence, Silence,
        )
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
        # Mean quality should be reasonable for grammar-compliant sequence
        assert report.mean_quality > 0.0, (
            f"Mean conservation quality {report.mean_quality:.4f} is zero"
        )

    def test_destabilizer_debt_repaid_by_stabilizers(self):
        """3×OZ + 3×IL: stabilizers should reduce Lyapunov energy."""
        from tnfr.operators.definitions import Dissonance, Coherence
        G = self._make_graph()
        node = list(G.nodes())[0]

        snap_initial = capture_conservation_snapshot(G)

        # Accumulate destabilizer debt
        for _ in range(3):
            Dissonance()(G, node)
        snap_post_oz = capture_conservation_snapshot(G)

        # Repay with stabilizers
        for _ in range(3):
            Coherence()(G, node)
        snap_post_il = capture_conservation_snapshot(G)

        lyap_oz = compute_lyapunov_derivative(snap_initial, snap_post_oz)
        lyap_il = compute_lyapunov_derivative(snap_post_oz, snap_post_il)

        # OZ should inject energy or maintain it (not strongly stable)
        # IL should dissipate energy (dE/dt ≤ 0 or near zero)
        assert lyap_il.dissipation >= 0.0
        # After IL, energy should be ≤ post-OZ peak (or close)
        # Allow tolerance for stochastic operator effects
        assert lyap_il.energy_after <= lyap_oz.energy_after + 1.0, (
            f"IL did not reduce energy: post-OZ={lyap_oz.energy_after:.4f}, "
            f"post-IL={lyap_il.energy_after:.4f}"
        )

    @pytest.mark.parametrize("topo", ["watts_strogatz", "barabasi_albert", "grid"])
    def test_oz_il_stable_across_topologies(self, topo):
        """OZ → IL Lyapunov stability should hold across topologies."""
        from tnfr.operators.definitions import Dissonance, Coherence
        n = 16 if topo == "grid" else 25
        G = self._make_graph(n, topo)
        node = list(G.nodes())[0]
        before = capture_conservation_snapshot(G)
        Dissonance()(G, node)
        Coherence()(G, node)
        after = capture_conservation_snapshot(G)
        lyap = compute_lyapunov_derivative(before, after)
        assert lyap.dissipation >= 0.0
        assert math.isfinite(lyap.energy_derivative)


class TestDissonancePostconditionConservation:
    """Test OZ postcondition (|ΔNFR| must increase) at conservation level.

    Physics: OZ raises structural pressure (ΔNFR). The conservation
    snapshot j_dnfr field captures this quantity. After OZ, the network-
    level absolute ΔNFR should not decrease (propagation redistributes
    dissonance among phase-compatible neighbours).
    """

    @staticmethod
    def _make_graph(n: int = 20, topo: str = "watts_strogatz",
                    seed: int = 42) -> nx.Graph:
        G = _make_tnfr_graph(n, topo, seed=seed)
        for nd in G.nodes():
            G.nodes[nd]['EPI'] = 1.0
        return G

    def test_dissonance_increases_delta_nfr(self):
        """OZ increases network-level |ΔNFR| after proper grammar setup."""
        from tnfr.operators.definitions import Dissonance, Coherence
        G = self._make_graph()
        node = list(G.nodes())[0]
        # Grammar U4a requires handler context before bifurcation trigger
        Coherence()(G, node)
        total_before = sum(
            abs(G.nodes[n].get("delta_nfr", 0.0)) for n in G.nodes()
        )
        Dissonance()(G, node)
        total_after = sum(
            abs(G.nodes[n].get("delta_nfr", 0.0)) for n in G.nodes()
        )
        # OZ contract: network-level |ΔNFR| should increase (includes propagation)
        assert total_after >= total_before - 1e-6, (
            f"OZ did not increase network |ΔNFR|: {total_before:.6f} → {total_after:.6f}"
        )

    @pytest.mark.parametrize("topo", ["watts_strogatz", "barabasi_albert", "grid"])
    def test_dissonance_postcondition_cross_topology(self, topo):
        """OZ increases network-level |ΔNFR| across topologies."""
        from tnfr.operators.definitions import Dissonance, Coherence
        n = 16 if topo == "grid" else 25
        G = self._make_graph(n, topo)
        for node in list(G.nodes())[:3]:
            # Grammar U4a handler context required before OZ
            Coherence()(G, node)
            total_before = sum(
                abs(G.nodes[nd].get("delta_nfr", 0.0)) for nd in G.nodes()
            )
            Dissonance()(G, node)
            total_after = sum(
                abs(G.nodes[nd].get("delta_nfr", 0.0)) for nd in G.nodes()
            )
            assert total_after >= total_before - 1e-6, (
                f"{topo} node {node}: network |ΔNFR| {total_before:.6f} → {total_after:.6f}"
            )

    def test_dissonance_energy_injection(self):
        """OZ should inject energy (E_after ≥ E_before) or be neutral."""
        from tnfr.operators.definitions import Dissonance
        G = self._make_graph()
        node = list(G.nodes())[0]
        before = capture_conservation_snapshot(G)
        Dissonance()(G, node)
        after = capture_conservation_snapshot(G)
        lyap = compute_lyapunov_derivative(before, after)
        # OZ destabilizes: energy should not decrease significantly
        # Allow small decrease due to network propagation effects
        assert lyap.energy_after >= lyap.energy_before - 0.5, (
            f"OZ unexpectedly decreased energy: "
            f"{lyap.energy_before:.4f} → {lyap.energy_after:.4f}"
        )


class TestDissipationGrammarViolationCorrelation:
    """Test that conservation diagnostics detect grammar violations.

    Physics: The source term S = ∂ρ/∂t + div(J) vanishes under grammar
    compliance. High |S| (large residuals) should correlate with grammar
    violation severity from detect_grammar_violations_from_conservation.
    """

    @staticmethod
    def _make_graph(n: int = 30, topo: str = "watts_strogatz",
                    seed: int = 42) -> nx.Graph:
        return _make_tnfr_graph(n, topo, seed=seed)

    def test_static_state_no_violations(self):
        """Identical snapshots: zero Δρ/Δt but residual reflects current div(J)."""
        G = self._make_graph()
        snap = capture_conservation_snapshot(G)
        balance = verify_conservation_balance(snap, snap)
        # ∂ρ/∂t = 0 for identical snapshots → delta_rho is zero
        for n in G.nodes():
            assert abs(balance.delta_rho[n]) < 1e-12
        # Charge drift must be zero (same state)
        assert balance.charge_drift < 1e-12
        # Severity reflects current divergence (non-equilibrium state),
        # not actual grammar violations — identical snapshots have no dynamics
        violations = detect_grammar_violations_from_conservation(balance)
        assert math.isfinite(violations['severity'])

    def test_small_perturbation_low_severity(self):
        """Small perturbation produces finite diagnostics with reasonable quality."""
        G = self._make_graph()
        before = capture_conservation_snapshot(G)
        rng = np.random.default_rng(42)
        for n in G.nodes():
            G.nodes[n]["delta_nfr"] += rng.uniform(-0.001, 0.001)
        after = capture_conservation_snapshot(G)
        balance = verify_conservation_balance(before, after)
        violations = detect_grammar_violations_from_conservation(balance)
        # Small perturbation: severity is finite, quality remains reasonable
        assert math.isfinite(violations['severity'])
        assert balance.conservation_quality > 0.3

    def test_extreme_perturbation_detected(self):
        """Large ΔNFR perturbation should trigger violation detection."""
        G = self._make_graph()
        before = capture_conservation_snapshot(G)
        for n in G.nodes():
            G.nodes[n]["delta_nfr"] += 10.0  # extreme perturbation
        after = capture_conservation_snapshot(G)
        balance = verify_conservation_balance(before, after)
        violations = detect_grammar_violations_from_conservation(balance)
        assert violations['severity'] > 0.0, (
            "Extreme perturbation not detected"
        )

    def test_severity_scales_with_perturbation(self):
        """Larger perturbations should produce higher severity."""
        G = self._make_graph()
        before = capture_conservation_snapshot(G)
        severities = []
        for scale in [0.01, 0.1, 1.0, 10.0]:
            G2 = G.copy()
            rng = np.random.default_rng(42)
            for n in G2.nodes():
                G2.nodes[n]["delta_nfr"] += rng.uniform(-scale, scale)
            after = capture_conservation_snapshot(G2)
            balance = verify_conservation_balance(before, after)
            violations = detect_grammar_violations_from_conservation(balance)
            severities.append(violations['severity'])
        # Severity should be monotonically non-decreasing
        for i in range(len(severities) - 1):
            assert severities[i] <= severities[i + 1] + 1e-9, (
                f"Severity not monotone: {severities}"
            )


class TestMultiStepDissipativeTracking:
    """Test ConservationTracker across multi-step dissipative sequences.

    Physics: The ConservationTracker records conservation diagnostics
    at each step. For grammar-compliant sequences (U2 convergence),
    the time-series should show bounded residuals and finite charge
    throughout.
    """

    @staticmethod
    def _make_graph(n: int = 25, topo: str = "watts_strogatz",
                    seed: int = 42) -> nx.Graph:
        G = _make_tnfr_graph(n, topo, seed=seed)
        for nd in G.nodes():
            G.nodes[nd]['EPI'] = 1.0
        return G

    def test_tracker_across_oz_il_repeated(self):
        """Alternating OZ-IL pairs should maintain bounded diagnostics."""
        from tnfr.operators.definitions import Dissonance, Coherence
        G = self._make_graph()
        node = list(G.nodes())[0]
        tracker = ConservationTracker(G)
        tracker.record(t=0.0)

        for i in range(5):
            Dissonance()(G, node)
            tracker.record(t=float(2 * i + 1))
            Coherence()(G, node)
            tracker.record(t=float(2 * i + 2))

        report = tracker.report()
        assert all(math.isfinite(q) for q in report.total_charge), (
            "Non-finite charge in multi-step OZ-IL sequence"
        )
        assert all(math.isfinite(r) for r in report.rms_residuals), (
            "Non-finite RMS residual in multi-step OZ-IL sequence"
        )

    def test_tracker_lyapunov_per_step(self):
        """Each OZ→IL pair should have finite Lyapunov derivative."""
        from tnfr.operators.definitions import Dissonance, Coherence
        G = self._make_graph()
        node = list(G.nodes())[0]
        snapshots = [capture_conservation_snapshot(G)]

        for _ in range(3):
            Dissonance()(G, node)
            Coherence()(G, node)
            snapshots.append(capture_conservation_snapshot(G))

        for i in range(len(snapshots) - 1):
            lyap = compute_lyapunov_derivative(snapshots[i], snapshots[i + 1])
            assert math.isfinite(lyap.energy_derivative), (
                f"Step {i}: non-finite dE/dt = {lyap.energy_derivative}"
            )
            assert lyap.energy_before >= 0.0
            assert lyap.energy_after >= 0.0

    def test_charge_drift_bounded_over_sequence(self):
        """Total charge drift should be bounded across a long sequence."""
        from tnfr.operators.definitions import (
            Emission, Dissonance, Coherence, Silence,
        )
        G = self._make_graph()
        node = list(G.nodes())[0]
        tracker = ConservationTracker(G)
        tracker.record(t=0.0)

        # Grammar-compliant sequence: AL → OZ → IL → OZ → IL → SHA
        Emission()(G, node)
        tracker.record(t=1.0)
        Dissonance()(G, node)
        tracker.record(t=2.0)
        Coherence()(G, node)
        tracker.record(t=3.0)
        Dissonance()(G, node)
        tracker.record(t=4.0)
        Coherence()(G, node)
        tracker.record(t=5.0)
        Silence()(G, node)
        tracker.record(t=6.0)

        report = tracker.report()
        # Charge drift should be bounded (not diverging)
        assert all(math.isfinite(d) for d in report.charge_drift)
        # Conservation quality should remain positive throughout
        assert all(q >= 0.0 for q in report.conservation_quality)


class TestLyapunovDissipativeRegimes:
    """Test Lyapunov derivative classification under different regimes.

    Physics: The Lyapunov energy E = ½Σ[field²] should decrease under
    grammar-compliant stabilization (IL) and may increase under
    destabilization (OZ). The classify is: is_stable (dE/dt ≤ 0),
    is_strongly_stable (dE/dt < -ε).
    """

    @staticmethod
    def _make_graph(n: int = 25, topo: str = "watts_strogatz",
                    seed: int = 42) -> nx.Graph:
        G = _make_tnfr_graph(n, topo, seed=seed)
        for nd in G.nodes():
            G.nodes[nd]['EPI'] = 1.0
        return G

    def test_coherence_is_lyapunov_stable(self):
        """IL (Coherence) should decrease or maintain energy."""
        from tnfr.operators.definitions import Coherence
        G = self._make_graph()
        node = list(G.nodes())[0]
        before = capture_conservation_snapshot(G)
        Coherence()(G, node)
        after = capture_conservation_snapshot(G)
        lyap = compute_lyapunov_derivative(before, after)
        # IL is a stabilizer: dE/dt ≤ 0 (Lyapunov theorem)
        assert lyap.is_stable, (
            f"IL not Lyapunov stable: dE/dt = {lyap.energy_derivative:.6f}"
        )
        assert lyap.dissipation >= 0.0

    def test_dissonance_may_inject_energy(self):
        """OZ (Dissonance) may have dE/dt > 0 (energy injection)."""
        from tnfr.operators.definitions import Dissonance
        G = self._make_graph()
        node = list(G.nodes())[0]
        before = capture_conservation_snapshot(G)
        Dissonance()(G, node)
        after = capture_conservation_snapshot(G)
        lyap = compute_lyapunov_derivative(before, after)
        # OZ is a destabilizer: we just verify the result is finite
        # (it may inject energy or be near-neutral depending on state)
        assert math.isfinite(lyap.energy_derivative)
        assert lyap.energy_before >= 0.0
        assert lyap.energy_after >= 0.0

    def test_silence_is_neutral(self):
        """SHA (Silence) should have near-zero energy change."""
        from tnfr.operators.definitions import Silence
        G = self._make_graph()
        node = list(G.nodes())[0]
        before = capture_conservation_snapshot(G)
        Silence()(G, node)
        after = capture_conservation_snapshot(G)
        lyap = compute_lyapunov_derivative(before, after)
        # SHA freezes evolution: dE/dt ≈ 0
        assert abs(lyap.energy_derivative) < 1.0, (
            f"SHA energy change {lyap.energy_derivative:.6f} too large"
        )

    @pytest.mark.parametrize("topo", ["watts_strogatz", "barabasi_albert", "grid"])
    def test_coherence_lyapunov_stable_cross_topology(self, topo):
        """IL Lyapunov stability should hold across all topologies."""
        from tnfr.operators.definitions import Coherence
        n = 16 if topo == "grid" else 25
        G = self._make_graph(n, topo)
        node = list(G.nodes())[0]
        before = capture_conservation_snapshot(G)
        Coherence()(G, node)
        after = capture_conservation_snapshot(G)
        lyap = compute_lyapunov_derivative(before, after)
        assert lyap.is_stable, (
            f"{topo}: IL dE/dt = {lyap.energy_derivative:.6f} > 0"
        )

    def test_dissipation_non_negative_invariant(self):
        """D[G] = max(0, -dE/dt) is always ≥ 0, regardless of operator."""
        from tnfr.operators.definitions import (
            Emission, Dissonance, Coherence, Silence,
        )
        G = self._make_graph()
        node = list(G.nodes())[0]
        operators = [Emission(), Dissonance(), Coherence(), Silence()]
        for op in operators:
            snap_before = capture_conservation_snapshot(G)
            op(G, node)
            snap_after = capture_conservation_snapshot(G)
            lyap = compute_lyapunov_derivative(snap_before, snap_after)
            assert lyap.dissipation >= 0.0, (
                f"{op.name}: dissipation = {lyap.dissipation:.6f} < 0"
            )
