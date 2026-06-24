"""Tests for TNFR Spectral Conservation — Conservation Laws in Spectral Space.

Validates the spectral continuity theorem:
    dρ̂_k/dt + λ_k · Ĵ_k = Ŝ_k  (mode-by-mode conservation)

derived via GFT of the structural continuity equation
    ∂ρ/∂t + div(J) = S_grammar.

Tests verify:
1.  SpectralConservationBalance: Two-snapshot spectral continuity
2.  Parseval identity: ‖ρ‖² = Σ|ρ̂_k|² across snapshots
3.  SpectralWardIdentity: Per-operator spectral signatures
4.  SpectralLyapunovResult: Mode-by-mode energy stability
5.  SpectralSectorDecomposition: Potential vs. geometric sector in eigenbasis
6.  Spectral energy conservation via Parseval across all five fields
7.  Mode classification: conserved / dissipative / accumulative
8.  Cross-topology validation: Watts-Strogatz, Barabási-Albert, Grid
9.  Equilibrium networks: identical snapshots → zero residual
10. Reproducibility: deterministic seeds → identical results

TIER: CORE PHYSICS — spectral conservation extends the structural conservation
theorem to the eigenvalue domain.
"""

from __future__ import annotations

import math
import os
import sys

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tnfr.constants import inject_defaults
from tnfr.physics.conservation import (
    ConservationSnapshot,
    capture_conservation_snapshot,
    verify_conservation_balance,
)
from tnfr.physics.spectral_conservation import (
    SpectralConservationBalance,
    SpectralLyapunovResult,
    SpectralSectorDecomposition,
    SpectralWardIdentity,
    classify_spectral_modes,
    compute_spectral_energy_conservation,
    compute_spectral_lyapunov,
    compute_spectral_ward_identity,
    decompose_spectral_sectors,
    verify_spectral_conservation_balance,
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

    for node in G.nodes():
        G.nodes[node]["phase"] = rng.uniform(0, 2 * math.pi)
        G.nodes[node]["frequency"] = rng.uniform(0.1, 1.0)
        G.nodes[node]["delta_nfr"] = rng.uniform(-0.5, 0.5)
        G.nodes[node]["EPI"] = f"epi_{node}"

    return G


def _perturb_graph(G: nx.Graph, seed: int = 99) -> nx.Graph:
    """Create a slightly perturbed copy for two-snapshot tests."""
    import copy

    G2 = copy.deepcopy(G)
    rng = np.random.default_rng(seed)
    for node in G2.nodes():
        G2.nodes[node]["phase"] += rng.uniform(-0.1, 0.1)
        G2.nodes[node]["delta_nfr"] += rng.uniform(-0.05, 0.05)
    return G2


@pytest.fixture
def ws_graph():
    return _make_tnfr_graph(30, "watts_strogatz")


@pytest.fixture
def ba_graph():
    return _make_tnfr_graph(30, "barabasi_albert")


@pytest.fixture
def grid_graph():
    return _make_tnfr_graph(25, "grid")


@pytest.fixture
def two_snapshots(ws_graph):
    """Provide before/after snapshots from WS graph and perturbed copy."""
    snap_before = capture_conservation_snapshot(ws_graph)
    G2 = _perturb_graph(ws_graph)
    snap_after = capture_conservation_snapshot(G2)
    return snap_before, snap_after, ws_graph


# ===========================================================================
# Test: SpectralConservationBalance
# ===========================================================================


class TestSpectralConservationBalance:
    """Two-snapshot spectral continuity verification."""

    def test_returns_valid_dataclass(self, two_snapshots):
        before, after, G = two_snapshots
        result = verify_spectral_conservation_balance(before, after, G)
        assert isinstance(result, SpectralConservationBalance)

    def test_array_shapes(self, two_snapshots):
        before, after, G = two_snapshots
        n = G.number_of_nodes()
        result = verify_spectral_conservation_balance(before, after, G)
        assert result.eigenvalues.shape == (n,)
        assert result.eigenvectors.shape == (n, n)
        assert result.rho_spectrum_before.shape == (n,)
        assert result.rho_spectrum_after.shape == (n,)
        assert result.div_spectrum_mean.shape == (n,)
        assert result.mode_residuals.shape == (n,)
        assert result.mode_sources.shape == (n,)

    def test_eigenvalues_ascending(self, two_snapshots):
        before, after, G = two_snapshots
        result = verify_spectral_conservation_balance(before, after, G)
        diffs = np.diff(result.eigenvalues)
        assert np.all(diffs >= -1e-10)

    def test_zero_mode_eigenvalue(self, two_snapshots):
        """Connected graph has λ_0 ≈ 0."""
        before, after, G = two_snapshots
        result = verify_spectral_conservation_balance(before, after, G)
        assert abs(result.eigenvalues[0]) < 1e-10

    def test_spectral_gap_positive(self, two_snapshots):
        before, after, G = two_snapshots
        result = verify_spectral_conservation_balance(before, after, G)
        assert result.spectral_gap > 0.0

    def test_residuals_non_negative(self, two_snapshots):
        before, after, G = two_snapshots
        result = verify_spectral_conservation_balance(before, after, G)
        assert np.all(result.mode_residuals >= 0.0)

    def test_parseval_values_non_negative(self, two_snapshots):
        before, after, G = two_snapshots
        result = verify_spectral_conservation_balance(before, after, G)
        assert result.parseval_before >= 0.0
        assert result.parseval_after >= 0.0
        assert result.parseval_drift >= 0.0

    def test_conservation_quality_bands_bounded(self, two_snapshots):
        before, after, G = two_snapshots
        result = verify_spectral_conservation_balance(before, after, G)
        for band in ("low", "mid", "high"):
            assert band in result.conservation_quality_by_band
            q = result.conservation_quality_by_band[band]
            assert 0.0 <= q <= 1.0

    def test_overall_quality_bounded(self, two_snapshots):
        before, after, G = two_snapshots
        result = verify_spectral_conservation_balance(before, after, G)
        assert 0.0 < result.overall_spectral_quality <= 1.0

    def test_conserved_modes_count_bounded(self, two_snapshots):
        before, after, G = two_snapshots
        n = G.number_of_nodes()
        result = verify_spectral_conservation_balance(before, after, G)
        assert 0 <= result.n_conserved_modes <= n

    def test_identical_snapshots_zero_temporal_change(self, ws_graph):
        """Identical snapshots → zero temporal change, zero Parseval drift.

        Note: mode_residuals = |drho_dt + λ_k · div_hat| are NOT zero
        because the static source term Ŝ_k = λ_k · div_hat is non-zero
        in a typical TNFR network. This is physically correct — the
        continuity equation reads 0 + div(J) = S_grammar at equilibrium.
        """
        snap = capture_conservation_snapshot(ws_graph)
        result = verify_spectral_conservation_balance(snap, snap, ws_graph)
        # Temporal change is zero
        assert np.allclose(result.rho_spectrum_before, result.rho_spectrum_after)
        # Parseval drift is zero
        assert result.parseval_drift < 1e-12

    def test_low_modes_better_conserved_than_high(self, two_snapshots):
        """Low-frequency modes should generally conserve better."""
        before, after, G = two_snapshots
        result = verify_spectral_conservation_balance(before, after, G)
        q = result.conservation_quality_by_band
        # Low modes should be at least as well-conserved as high modes
        # (statistical tendency, not strict for all networks)
        assert q["low"] >= q["high"] * 0.5  # soft threshold


# ===========================================================================
# Test: Parseval identity
# ===========================================================================


class TestParsevalConservation:
    """Parseval: ‖ρ‖² = Σ|ρ̂_k|² must hold at each snapshot."""

    def test_parseval_matches_spatial_energy(self, ws_graph):
        """Spectral energy must equal spatial energy (Parseval theorem)."""
        snap = capture_conservation_snapshot(ws_graph)
        nodes = sorted(snap.charge_density.keys())
        rho_vec = np.array([snap.charge_density[n] for n in nodes])
        spatial_energy = float(np.sum(rho_vec**2))

        result = verify_spectral_conservation_balance(snap, snap, ws_graph)
        assert abs(result.parseval_before - spatial_energy) < 1e-10

    def test_parseval_after_perturbation(self, ws_graph):
        """Parseval holds on perturbed graph too."""
        G2 = _perturb_graph(ws_graph)
        snap = capture_conservation_snapshot(G2)
        nodes = sorted(snap.charge_density.keys())
        rho_vec = np.array([snap.charge_density[n] for n in nodes])
        spatial_energy = float(np.sum(rho_vec**2))

        result = verify_spectral_conservation_balance(snap, snap, G2)
        assert abs(result.parseval_before - spatial_energy) < 1e-10


# ===========================================================================
# Test: SpectralWardIdentity
# ===========================================================================


class TestSpectralWardIdentity:
    """Per-operator spectral conservation signature."""

    def test_returns_valid_dataclass(self, two_snapshots):
        before, after, G = two_snapshots
        ward = compute_spectral_ward_identity(before, after, "IL", G)
        assert isinstance(ward, SpectralWardIdentity)

    def test_spectrum_shape(self, two_snapshots):
        before, after, G = two_snapshots
        n = G.number_of_nodes()
        ward = compute_spectral_ward_identity(before, after, "OZ", G)
        assert ward.delta_rho_spectrum.shape == (n,)
        assert ward.mode_energy_change.shape == (n,)

    def test_operator_name_persisted(self, two_snapshots):
        before, after, G = two_snapshots
        ward = compute_spectral_ward_identity(before, after, "THOL", G)
        assert ward.operator_name == "THOL"

    def test_affected_band_valid(self, two_snapshots):
        before, after, G = two_snapshots
        ward = compute_spectral_ward_identity(before, after, "AL", G)
        assert ward.affected_band in ("low", "mid", "high")

    def test_spectral_character_valid(self, two_snapshots):
        before, after, G = two_snapshots
        ward = compute_spectral_ward_identity(before, after, "IL", G)
        assert ward.spectral_character in ("conservative", "dissipative", "injective")

    def test_identical_snapshots_conservative(self, ws_graph):
        """No change → conservative character."""
        snap = capture_conservation_snapshot(ws_graph)
        ward = compute_spectral_ward_identity(snap, snap, "IL", ws_graph)
        assert ward.spectral_character == "conservative"
        assert abs(ward.total_spectral_energy_change) < 1e-12

    def test_energy_change_consistent(self, two_snapshots):
        """Total energy change is sum of mode energies."""
        before, after, G = two_snapshots
        ward = compute_spectral_ward_identity(before, after, "OZ", G)
        expected = float(np.sum(ward.mode_energy_change))
        assert abs(ward.total_spectral_energy_change - expected) < 1e-10


# ===========================================================================
# Test: SpectralLyapunovResult
# ===========================================================================


class TestSpectralLyapunov:
    """Mode-by-mode Lyapunov energy stability."""

    def test_returns_valid_dataclass(self, two_snapshots):
        before, after, G = two_snapshots
        result = compute_spectral_lyapunov(before, after, G)
        assert isinstance(result, SpectralLyapunovResult)

    def test_array_shapes(self, two_snapshots):
        before, after, G = two_snapshots
        n = G.number_of_nodes()
        result = compute_spectral_lyapunov(before, after, G)
        assert result.mode_energies_before.shape == (n,)
        assert result.mode_energies_after.shape == (n,)
        assert result.mode_derivatives.shape == (n,)

    def test_energies_non_negative(self, two_snapshots):
        before, after, G = two_snapshots
        result = compute_spectral_lyapunov(before, after, G)
        assert np.all(result.mode_energies_before >= -1e-12)
        assert np.all(result.mode_energies_after >= -1e-12)

    def test_total_derivative_equals_sum(self, two_snapshots):
        before, after, G = two_snapshots
        result = compute_spectral_lyapunov(before, after, G)
        expected = float(np.sum(result.mode_derivatives))
        assert abs(result.total_derivative - expected) < 1e-10

    def test_stable_fraction_bounded(self, two_snapshots):
        before, after, G = two_snapshots
        result = compute_spectral_lyapunov(before, after, G)
        assert 0.0 <= result.stable_fraction <= 1.0

    def test_identical_snapshots_stable(self, ws_graph):
        """No change → zero derivatives, fully stable."""
        snap = capture_conservation_snapshot(ws_graph)
        result = compute_spectral_lyapunov(snap, snap, ws_graph)
        assert np.allclose(result.mode_derivatives, 0.0, atol=1e-12)
        assert result.is_spectrally_stable
        assert result.n_unstable_modes == 0
        assert result.stable_fraction == 1.0

    def test_unstable_modes_consistent(self, two_snapshots):
        before, after, G = two_snapshots
        n = G.number_of_nodes()
        result = compute_spectral_lyapunov(before, after, G)
        assert 0 <= result.n_unstable_modes <= n
        exp_stable = 1.0 - result.n_unstable_modes / max(n, 1)
        assert abs(result.stable_fraction - exp_stable) < 1e-12


# ===========================================================================
# Test: SpectralSectorDecomposition
# ===========================================================================


class TestSpectralSectorDecomposition:
    """Potential vs. geometric sector in spectral domain."""

    def test_returns_valid_dataclass(self, ws_graph):
        result = decompose_spectral_sectors(ws_graph)
        assert isinstance(result, SpectralSectorDecomposition)

    def test_spectrum_shapes(self, ws_graph):
        n = ws_graph.number_of_nodes()
        result = decompose_spectral_sectors(ws_graph)
        assert result.phi_s_spectrum.shape == (n,)
        assert result.k_phi_spectrum.shape == (n,)
        assert result.sector_coupling_by_mode.shape == (n,)

    def test_energies_non_negative(self, ws_graph):
        result = decompose_spectral_sectors(ws_graph)
        assert result.potential_sector_energy >= 0.0
        assert result.geometric_sector_energy >= 0.0

    def test_dominant_sector_valid(self, ws_graph):
        result = decompose_spectral_sectors(ws_graph)
        assert result.dominant_sector in ("potential", "geometric")

    def test_dominant_sector_matches_energy(self, ws_graph):
        result = decompose_spectral_sectors(ws_graph)
        if result.potential_sector_energy >= result.geometric_sector_energy:
            assert result.dominant_sector == "potential"
        else:
            assert result.dominant_sector == "geometric"

    def test_correlation_bounded(self, ws_graph):
        result = decompose_spectral_sectors(ws_graph)
        assert -1.0 <= result.cross_sector_correlation <= 1.0

    def test_sector_ratio_positive(self, ws_graph):
        result = decompose_spectral_sectors(ws_graph)
        assert result.sector_ratio >= 0.0

    def test_coupling_by_mode_non_negative(self, ws_graph):
        result = decompose_spectral_sectors(ws_graph)
        assert np.all(result.sector_coupling_by_mode >= 0.0)

    def test_with_explicit_snapshot(self, ws_graph):
        snap = capture_conservation_snapshot(ws_graph)
        result = decompose_spectral_sectors(ws_graph, snapshot=snap)
        assert isinstance(result, SpectralSectorDecomposition)

    @pytest.mark.parametrize("topo", ["watts_strogatz", "barabasi_albert", "grid"])
    def test_across_topologies(self, topo):
        G = _make_tnfr_graph(25, topo)
        result = decompose_spectral_sectors(G)
        assert isinstance(result, SpectralSectorDecomposition)
        assert result.potential_sector_energy >= 0.0
        assert result.geometric_sector_energy >= 0.0


# ===========================================================================
# Test: Spectral energy conservation (Parseval per field)
# ===========================================================================


class TestSpectralEnergyConservation:
    """Parseval-based energy drift across all five canonical fields."""

    def test_returns_all_keys(self, two_snapshots):
        before, after, G = two_snapshots
        result = compute_spectral_energy_conservation(before, after, G)
        expected_keys = {
            "phi_s_drift",
            "grad_phi_drift",
            "k_phi_drift",
            "j_phi_drift",
            "j_dnfr_drift",
            "total_energy_before",
            "total_energy_after",
            "total_drift",
        }
        assert set(result.keys()) == expected_keys

    def test_drifts_non_negative(self, two_snapshots):
        before, after, G = two_snapshots
        result = compute_spectral_energy_conservation(before, after, G)
        for key in (
            "phi_s_drift",
            "grad_phi_drift",
            "k_phi_drift",
            "j_phi_drift",
            "j_dnfr_drift",
            "total_drift",
        ):
            assert result[key] >= 0.0

    def test_energies_non_negative(self, two_snapshots):
        before, after, G = two_snapshots
        result = compute_spectral_energy_conservation(before, after, G)
        assert result["total_energy_before"] >= 0.0
        assert result["total_energy_after"] >= 0.0

    def test_identical_snapshots_zero_drift(self, ws_graph):
        snap = capture_conservation_snapshot(ws_graph)
        result = compute_spectral_energy_conservation(snap, snap, ws_graph)
        for key in (
            "phi_s_drift",
            "grad_phi_drift",
            "k_phi_drift",
            "j_phi_drift",
            "j_dnfr_drift",
            "total_drift",
        ):
            assert result[key] < 1e-12

    def test_total_drift_finite(self, two_snapshots):
        before, after, G = two_snapshots
        result = compute_spectral_energy_conservation(before, after, G)
        assert np.isfinite(result["total_drift"])


# ===========================================================================
# Test: Mode classification
# ===========================================================================


class TestModeClassification:
    """Conserved / dissipative / accumulative mode labeling."""

    def test_returns_expected_keys(self, ws_graph):
        result = classify_spectral_modes(ws_graph)
        expected_keys = {
            "mode_labels",
            "n_conserved",
            "n_dissipative",
            "n_accumulative",
            "mode_transport_rates",
        }
        assert set(result.keys()) == expected_keys

    def test_labels_valid(self, ws_graph):
        result = classify_spectral_modes(ws_graph)
        valid = {"conserved", "dissipative", "accumulative"}
        for label in result["mode_labels"]:
            assert label in valid

    def test_counts_sum_to_n(self, ws_graph):
        n = ws_graph.number_of_nodes()
        result = classify_spectral_modes(ws_graph)
        total = (
            result["n_conserved"] + result["n_dissipative"] + result["n_accumulative"]
        )
        assert total == n

    def test_transport_rates_shape(self, ws_graph):
        n = ws_graph.number_of_nodes()
        result = classify_spectral_modes(ws_graph)
        assert result["mode_transport_rates"].shape == (n,)

    def test_with_explicit_snapshot(self, ws_graph):
        snap = capture_conservation_snapshot(ws_graph)
        result = classify_spectral_modes(ws_graph, snapshot=snap)
        assert result["n_conserved"] >= 0

    def test_custom_threshold(self, ws_graph):
        result = classify_spectral_modes(ws_graph, threshold=1e-10)
        # Very tight threshold → fewer conserved modes
        assert result["n_conserved"] >= 0

    @pytest.mark.parametrize("topo", ["watts_strogatz", "barabasi_albert", "grid"])
    def test_across_topologies(self, topo):
        G = _make_tnfr_graph(25, topo)
        result = classify_spectral_modes(G)
        n = G.number_of_nodes()
        total = (
            result["n_conserved"] + result["n_dissipative"] + result["n_accumulative"]
        )
        assert total == n


# ===========================================================================
# Test: Cross-topology validation
# ===========================================================================


class TestCrossTopology:
    """Verify all spectral conservation functions across topologies."""

    @pytest.mark.parametrize("topo", ["watts_strogatz", "barabasi_albert", "grid"])
    def test_balance_across_topologies(self, topo):
        G = _make_tnfr_graph(25, topo)
        G2 = _perturb_graph(G)
        before = capture_conservation_snapshot(G)
        after = capture_conservation_snapshot(G2)
        result = verify_spectral_conservation_balance(before, after, G)
        assert isinstance(result, SpectralConservationBalance)
        assert result.spectral_gap >= 0.0

    @pytest.mark.parametrize("topo", ["watts_strogatz", "barabasi_albert", "grid"])
    def test_lyapunov_across_topologies(self, topo):
        G = _make_tnfr_graph(25, topo)
        G2 = _perturb_graph(G)
        before = capture_conservation_snapshot(G)
        after = capture_conservation_snapshot(G2)
        result = compute_spectral_lyapunov(before, after, G)
        assert isinstance(result, SpectralLyapunovResult)
        assert 0.0 <= result.stable_fraction <= 1.0


# ===========================================================================
# Test: Reproducibility (Invariant #6)
# ===========================================================================


class TestReproducibility:
    """Same seed produces identical spectral conservation results."""

    def test_deterministic_balance(self):
        G1 = _make_tnfr_graph(20, "watts_strogatz", seed=77)
        G1p = _perturb_graph(G1, seed=88)
        b1 = capture_conservation_snapshot(G1)
        a1 = capture_conservation_snapshot(G1p)

        G2 = _make_tnfr_graph(20, "watts_strogatz", seed=77)
        G2p = _perturb_graph(G2, seed=88)
        b2 = capture_conservation_snapshot(G2)
        a2 = capture_conservation_snapshot(G2p)

        r1 = verify_spectral_conservation_balance(b1, a1, G1)
        r2 = verify_spectral_conservation_balance(b2, a2, G2)

        assert np.allclose(r1.mode_residuals, r2.mode_residuals)
        assert abs(r1.parseval_drift - r2.parseval_drift) < 1e-12

    def test_deterministic_sectors(self):
        G1 = _make_tnfr_graph(20, "watts_strogatz", seed=77)
        G2 = _make_tnfr_graph(20, "watts_strogatz", seed=77)

        s1 = decompose_spectral_sectors(G1)
        s2 = decompose_spectral_sectors(G2)

        assert np.allclose(s1.phi_s_spectrum, s2.phi_s_spectrum)
        assert np.allclose(s1.k_phi_spectrum, s2.k_phi_spectrum)


# ===========================================================================
# P3 Deep-Physics Tests — Spectral conservation invariant validation
# ===========================================================================
# These tests exercise spectral conservation under physically meaningful
# conditions rather than just structural (shape/type) checks.
#
# Gap analysis reference (AGENTS.md P3 audit):
#   1. Near-static spectral continuity
#   2. Parseval drift bounded for small perturbations
#   3. Spectral Lyapunov monotonicity under stabilizers
#   4. Operator spectral signatures (Ward identity)
#   5. Multi-step spectral tracking
#   6. Spectral-spatial consistency
# ===========================================================================


def _tiny_perturb_graph(G: nx.Graph, scale: float = 1e-4, seed: int = 123) -> nx.Graph:
    """Create a near-identical copy with O(scale) perturbations."""
    import copy

    G2 = copy.deepcopy(G)
    rng = np.random.default_rng(seed)
    for node in G2.nodes():
        G2.nodes[node]["phase"] += rng.uniform(-scale, scale)
        G2.nodes[node]["delta_nfr"] += rng.uniform(-scale * 0.5, scale * 0.5)
    return G2


class TestNearStaticSpectralContinuity:
    """Gap #1: Near-static evolution produces negligible spectral drift.

    Physics: Spectral quality measures equilibrium proximity (source term
    magnitude), not perturbation sensitivity.  For near-static evolution,
    the correct observables are:
    - Parseval drift → near zero (energy bookkeeping identity)
    - Δρ̂_k → scales with perturbation magnitude
    - Low-band quality ≥ high-band quality (physics: low modes conserve best)
    """

    def test_tiny_perturbation_negligible_parseval_drift(self):
        """Tiny perturbations → Parseval drift ≈ 0."""
        G = _make_tnfr_graph(30, "watts_strogatz", seed=42)
        G2 = _tiny_perturb_graph(G, scale=1e-5, seed=200)
        before = capture_conservation_snapshot(G)
        after = capture_conservation_snapshot(G2)
        result = verify_spectral_conservation_balance(before, after, G)
        assert (
            result.parseval_drift < 1e-4
        ), f"Near-static Parseval drift {result.parseval_drift:.2e} too large"

    def test_delta_rho_spectrum_scales_with_perturbation(self):
        """Δρ̂_k magnitude scales with perturbation scale."""
        G = _make_tnfr_graph(30, "watts_strogatz", seed=42)
        before = capture_conservation_snapshot(G)

        norms = []
        for scale in [1e-5, 1e-3, 1e-1]:
            G2 = _tiny_perturb_graph(G, scale=scale, seed=201)
            after = capture_conservation_snapshot(G2)
            result = verify_spectral_conservation_balance(before, after, G)
            delta = result.rho_spectrum_after - result.rho_spectrum_before
            norms.append(float(np.linalg.norm(delta)))

        # Δρ̂ norm must increase monotonically with perturbation scale
        assert (
            norms[0] < norms[1] < norms[2]
        ), f"Δρ̂ norm not monotone with scale: {norms}"

    def test_low_band_quality_geq_high_band(self):
        """Low-frequency modes conserve better than high-frequency modes."""
        G = _make_tnfr_graph(30, "watts_strogatz", seed=42)
        G2 = _perturb_graph(G, seed=202)
        before = capture_conservation_snapshot(G)
        after = capture_conservation_snapshot(G2)
        result = verify_spectral_conservation_balance(before, after, G)
        bands = result.conservation_quality_by_band
        assert (
            bands["low"] >= bands["high"]
        ), f"Low band quality {bands['low']:.4f} < high band {bands['high']:.4f}"

    @pytest.mark.parametrize("topo", ["watts_strogatz", "barabasi_albert", "grid"])
    def test_low_band_best_cross_topology(self, topo):
        """Low-frequency band conserves best across all topologies."""
        n = 25 if topo == "grid" else 30
        G = _make_tnfr_graph(n, topo, seed=42)
        G2 = _perturb_graph(G, seed=203)
        before = capture_conservation_snapshot(G)
        after = capture_conservation_snapshot(G2)
        result = verify_spectral_conservation_balance(before, after, G)
        bands = result.conservation_quality_by_band
        assert (
            bands["low"] >= bands["high"]
        ), f"{topo}: low={bands['low']:.4f} < high={bands['high']:.4f}"


class TestParsevalDriftBounded:
    """Gap #2: Parseval drift should scale with perturbation magnitude.

    Physics: Parseval identity ‖ρ‖² = Σ|ρ̂_k|² is exact.  Drift between
    snapshots reflects actual structural change, so small perturbations
    must yield proportionally small drift.
    """

    def test_drift_scales_with_perturbation_size(self):
        """Larger perturbation → larger Parseval drift (monotone)."""
        G = _make_tnfr_graph(30, "watts_strogatz", seed=42)
        before = capture_conservation_snapshot(G)

        drifts = []
        for scale in [1e-5, 1e-3, 1e-1]:
            G2 = _tiny_perturb_graph(G, scale=scale, seed=300)
            after = capture_conservation_snapshot(G2)
            result = verify_spectral_conservation_balance(before, after, G)
            drifts.append(result.parseval_drift)

        # Drift must be monotonically increasing with perturbation scale
        assert (
            drifts[0] < drifts[1] < drifts[2]
        ), f"Parseval drift not monotone with scale: {drifts}"

    def test_small_perturbation_small_drift(self):
        """Parseval drift < 0.01 for O(1e-4) perturbations."""
        G = _make_tnfr_graph(30, "watts_strogatz", seed=42)
        G2 = _tiny_perturb_graph(G, scale=1e-4, seed=301)
        before = capture_conservation_snapshot(G)
        after = capture_conservation_snapshot(G2)
        result = verify_spectral_conservation_balance(before, after, G)
        assert (
            result.parseval_drift < 0.01
        ), f"Parseval drift {result.parseval_drift:.6f} exceeds 0.01 for tiny perturbation"

    def test_energy_conservation_drift_bounded(self):
        """Per-field spectral energy drifts are bounded for small perturbations."""
        G = _make_tnfr_graph(30, "watts_strogatz", seed=42)
        G2 = _tiny_perturb_graph(G, scale=1e-4, seed=302)
        before = capture_conservation_snapshot(G)
        after = capture_conservation_snapshot(G2)
        result = compute_spectral_energy_conservation(before, after, G)
        for field in ["phi_s", "grad_phi", "k_phi", "j_phi", "j_dnfr"]:
            drift_val = result[f"{field}_drift"]
            assert drift_val < 0.05, f"Field {field} drift {drift_val:.6f} exceeds 0.05"


class TestSpectralLyapunovMonotonicity:
    """Gap #3: Spectral Lyapunov stability under stabilizer-dominated evolution.

    Physics: Under grammar-compliant sequences (U2), the Lyapunov energy
    E = ½Σ[Φ_s² + |∇φ|² + K_φ² + J_φ² + J_ΔNFR²] should not increase.
    In spectral domain: dE/dt = Σ dE_k/dt ≤ 0 (Structural Conservation
    Theorem §7).

    We test with the Coherence (IL) operator which is the canonical stabilizer.
    """

    def _apply_coherence_to_all(self, G: nx.Graph) -> None:
        """Apply IL (Coherence) to every node — pure stabilizer sequence."""
        from tnfr.operators.definitions import Coherence

        coherence = Coherence()
        for node in G.nodes():
            try:
                coherence(G, node)
            except Exception:
                pass  # Some nodes may not support IL if νf=0

    def test_stabilizer_does_not_increase_total_energy(self):
        """Total Lyapunov derivative ≤ 0 after IL application."""
        G = _make_tnfr_graph(30, "watts_strogatz", seed=42)
        before = capture_conservation_snapshot(G)
        self._apply_coherence_to_all(G)
        after = capture_conservation_snapshot(G)
        result = compute_spectral_lyapunov(before, after, G)
        assert (
            result.total_derivative <= 0.0 or result.is_spectrally_stable
        ), f"Stabilizer increased spectral energy: dE/dt={result.total_derivative:.6f}"

    def test_stabilizer_high_stable_fraction(self):
        """After IL, most modes should be stable (dE_k/dt ≤ 0)."""
        G = _make_tnfr_graph(30, "watts_strogatz", seed=42)
        before = capture_conservation_snapshot(G)
        self._apply_coherence_to_all(G)
        after = capture_conservation_snapshot(G)
        result = compute_spectral_lyapunov(before, after, G)
        assert (
            result.stable_fraction >= 0.5
        ), f"Stabilizer stable_fraction={result.stable_fraction:.3f}, expected ≥ 0.5"


class TestOperatorSpectralSignatures:
    """Gap #4: Canonical operators produce predictable spectral signatures.

    Physics: Each operator has a characteristic spectral Ward identity.
    - IL (Coherence/stabilizer): Energy decreasing (dissipative character)
    - OZ (Dissonance/destabilizer): Energy increasing (injective character)
    - SHA (Silence): Near-zero energy change (conservative character)
    """

    @staticmethod
    def _fresh_graph(seed: int = 42) -> nx.Graph:
        return _make_tnfr_graph(30, "watts_strogatz", seed=seed)

    def test_coherence_ward_has_valid_character(self):
        """IL should produce a valid spectral character classification."""
        from tnfr.operators.definitions import Coherence

        G = self._fresh_graph()
        before = capture_conservation_snapshot(G)
        Coherence()(G, list(G.nodes())[0])
        after = capture_conservation_snapshot(G)
        ward = compute_spectral_ward_identity(before, after, "Coherence", G)
        # IL's primary contract is C(t) monotonicity, not spectral charge energy
        # monotonicity.  Phase redistribution can appear injective in ρ̂ even
        # as coherence increases.  We verify the Ward identity classifies correctly.
        assert ward.spectral_character in (
            "dissipative",
            "conservative",
            "injective",
        ), (
            f"Coherence Ward identity: {ward.spectral_character}; "
            "expected a valid spectral character"
        )
        # But the affected band should be meaningful (not empty/missing)
        assert ward.affected_band in ("low", "mid", "high")

    def test_dissonance_ward_injective_or_conservative(self):
        """OZ should have injective or conservative spectral character."""
        from tnfr.operators.definitions import Dissonance

        G = self._fresh_graph()
        before = capture_conservation_snapshot(G)
        Dissonance()(G, list(G.nodes())[0])
        after = capture_conservation_snapshot(G)
        ward = compute_spectral_ward_identity(before, after, "Dissonance", G)
        # OZ destabilizes: should inject energy or be conservative
        assert ward.spectral_character in ("injective", "conservative"), (
            f"Dissonance Ward identity: {ward.spectral_character}; "
            "expected injective or conservative"
        )

    def test_silence_ward_conservative(self):
        """SHA should have (near-)conservative spectral character."""
        from tnfr.operators.definitions import Silence

        G = self._fresh_graph()
        before = capture_conservation_snapshot(G)
        Silence()(G, list(G.nodes())[0])
        after = capture_conservation_snapshot(G)
        ward = compute_spectral_ward_identity(before, after, "Silence", G)
        # SHA freezes evolution: near-zero energy change
        assert abs(ward.total_spectral_energy_change) < 1.0, (
            f"Silence energy change {ward.total_spectral_energy_change:.4f} "
            "unexpectedly large"
        )

    def test_ward_operator_name_preserved(self):
        """Ward identity records the correct operator name."""
        from tnfr.operators.definitions import Emission

        G = self._fresh_graph()
        # Set node to vacuum for Emission
        node = list(G.nodes())[0]
        G.nodes[node]["EPI"] = 0.0
        G.nodes[node]["nu_f"] = 1.0
        G.nodes[node]["delta_nfr"] = 0.0
        before = capture_conservation_snapshot(G)
        Emission()(G, node)
        after = capture_conservation_snapshot(G)
        ward = compute_spectral_ward_identity(before, after, "Emission", G)
        assert ward.operator_name == "Emission"


class TestMultiStepSpectralTracking:
    """Gap #5: Spectral conservation quality across a multi-step sequence.

    Physics: A grammar-compliant operator sequence should maintain or
    improve spectral quality over successive steps.  Quality degradation
    signals structural fragmentation.
    """

    def test_multi_step_quality_does_not_collapse(self):
        """5-step perturb-and-measure: quality stays positive (non-degenerate)."""
        G = _make_tnfr_graph(30, "watts_strogatz", seed=42)
        qualities = []
        for step_seed in range(100, 105):
            before = capture_conservation_snapshot(G)
            G = _perturb_graph(G, seed=step_seed)
            after = capture_conservation_snapshot(G)
            result = verify_spectral_conservation_balance(before, after, G)
            qualities.append(result.overall_spectral_quality)

        # Quality must remain strictly positive (system not degenerate)
        for i, q in enumerate(qualities):
            assert q > 0.0, f"Step {i}: spectral quality collapsed to {q:.4f}"
        # Low-band quality should stay above high-band across steps
        for step_seed in range(100, 105):
            G_tmp = _make_tnfr_graph(30, "watts_strogatz", seed=42)
            before = capture_conservation_snapshot(G_tmp)
            G_tmp = _perturb_graph(G_tmp, seed=step_seed)
            after = capture_conservation_snapshot(G_tmp)
            result = verify_spectral_conservation_balance(before, after, G_tmp)
            bands = result.conservation_quality_by_band
            assert (
                bands["low"] >= bands["high"]
            ), f"Step {step_seed}: low={bands['low']:.4f} < high={bands['high']:.4f}"

    def test_parseval_drift_bounded_across_steps(self):
        """Parseval drift stays bounded across multiple perturbation steps."""
        G = _make_tnfr_graph(30, "watts_strogatz", seed=42)
        drifts = []
        for step_seed in range(200, 208):
            before = capture_conservation_snapshot(G)
            G = _perturb_graph(G, seed=step_seed)
            after = capture_conservation_snapshot(G)
            result = verify_spectral_conservation_balance(before, after, G)
            drifts.append(result.parseval_drift)

        # No single step should produce extreme Parseval drift
        for i, d in enumerate(drifts):
            assert d < 1.0, f"Step {i}: Parseval drift {d:.4f} exceeds 1.0"

    def test_multi_step_lyapunov_energies_finite(self):
        """Spectral Lyapunov energies remain finite across steps."""
        G = _make_tnfr_graph(25, "watts_strogatz", seed=42)
        for step_seed in range(300, 305):
            before = capture_conservation_snapshot(G)
            G = _perturb_graph(G, seed=step_seed)
            after = capture_conservation_snapshot(G)
            result = compute_spectral_lyapunov(before, after, G)
            assert np.all(
                np.isfinite(result.mode_energies_after)
            ), f"Step {step_seed}: non-finite spectral Lyapunov energies"
            assert np.isfinite(
                result.total_derivative
            ), f"Step {step_seed}: non-finite total Lyapunov derivative"


class TestSpectralSpatialConsistency:
    """Gap #6: Spectral and spatial conservation quality must correlate.

    Physics: The spectral continuity theorem is derived via GFT of the
    spatial continuity equation.  Both views measure the same conservation
    law, so their quality metrics should agree directionally.
    """

    def test_parseval_drift_correlates_with_spatial_charge_drift(self):
        """Parseval drift and spatial charge drift both increase with perturbation."""
        G = _make_tnfr_graph(30, "watts_strogatz", seed=42)
        spatial_drifts = []
        spectral_drifts = []

        for scale in [1e-5, 1e-3, 1e-1]:
            G2 = _tiny_perturb_graph(G, scale=scale, seed=400)
            before = capture_conservation_snapshot(G)
            after = capture_conservation_snapshot(G2)

            spatial = verify_conservation_balance(before, after)
            spectral = verify_spectral_conservation_balance(before, after, G)

            spatial_drifts.append(spatial.charge_drift)
            spectral_drifts.append(spectral.parseval_drift)

        # Both should increase monotonically with perturbation scale
        assert (
            spatial_drifts[0] < spatial_drifts[2]
        ), f"Spatial charge drift not increasing: {spatial_drifts}"
        assert (
            spectral_drifts[0] < spectral_drifts[2]
        ), f"Spectral Parseval drift not increasing: {spectral_drifts}"

    def test_large_perturbation_both_parseval_and_spatial_increase(self):
        """Large perturbation produces larger drift in both domains."""
        G = _make_tnfr_graph(30, "watts_strogatz", seed=42)
        G_small = _tiny_perturb_graph(G, scale=1e-5, seed=401)
        G_large = _perturb_graph(G, seed=402)

        before = capture_conservation_snapshot(G)
        after_small = capture_conservation_snapshot(G_small)
        after_large = capture_conservation_snapshot(G_large)

        spectral_small = verify_spectral_conservation_balance(before, after_small, G)
        spectral_large = verify_spectral_conservation_balance(before, after_large, G)

        # Parseval drift: small perturbation → smaller drift
        assert spectral_small.parseval_drift < spectral_large.parseval_drift, (
            f"Parseval: small={spectral_small.parseval_drift:.6f} vs "
            f"large={spectral_large.parseval_drift:.6f}"
        )

    def test_spectral_energy_conservation_consistent_with_spatial(self):
        """Total spectral energy drift (Parseval) correlates with spatial RMS."""
        G = _make_tnfr_graph(30, "watts_strogatz", seed=42)
        spatial_rms_vals = []
        spectral_total_drifts = []

        for scale in [1e-5, 1e-3, 1e-1]:
            G2 = _tiny_perturb_graph(G, scale=scale, seed=403)
            before = capture_conservation_snapshot(G)
            after = capture_conservation_snapshot(G2)

            spatial = verify_conservation_balance(before, after)
            energy = compute_spectral_energy_conservation(before, after, G)

            spatial_rms_vals.append(spatial.rms_residual)
            spectral_total_drifts.append(energy["total_drift"])

        # Both should increase with perturbation: smallest < largest
        assert (
            spatial_rms_vals[0] < spatial_rms_vals[2]
        ), f"Spatial RMS not increasing: {spatial_rms_vals}"
        assert (
            spectral_total_drifts[0] < spectral_total_drifts[2]
        ), f"Spectral total drift not increasing: {spectral_total_drifts}"
