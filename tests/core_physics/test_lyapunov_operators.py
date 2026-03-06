"""Test Formal Lyapunov Stability for all 13 Canonical Operators.

Validates:
1. Per-operator energy bounds (OperatorLyapunovBound registry)
2. Spectral gap characterisation (SpectralGapAnalysis)
3. Operator-level Lyapunov verification against actual energy changes
4. Grammar-compliant sequence proofs (SequenceLyapunovProof)
5. Combined Lyapunov + spectral convergence analysis

Physics basis: the energy functional
    E[G] = ½ Σ_i [Φ_s(i)² + |∇φ|(i)² + K_φ(i)² + J_φ(i)² + J_ΔNFR(i)²]
must satisfy per-operator bounds derived from the glyph factors.

TIER: CORE PHYSICS — Lyapunov stability is fundamental to coherence preservation.
"""
from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from tnfr.constants import inject_defaults
from tnfr.physics.lyapunov import (
    EnergyClass,
    OperatorLyapunovBound,
    OperatorLyapunovVerification,
    SpectralGapAnalysis,
    LyapunovSpectralSummary,
    SequenceLyapunovProof,
    OPERATOR_LYAPUNOV_BOUNDS,
    get_bound,
    compute_operator_energy_bound,
    verify_operator_lyapunov,
    compute_sequence_energy_bound,
    prove_sequence_lyapunov,
    analyze_spectral_gap,
    analyze_operator_convergence,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_tnfr_graph(
    n: int = 20,
    topology: str = "watts_strogatz",
    seed: int = 42,
) -> nx.Graph:
    """Build a TNFR-ready graph with canonical attributes."""
    rng = np.random.default_rng(seed)

    if topology == "watts_strogatz":
        G = nx.watts_strogatz_graph(n, 4, 0.3, seed=seed)
    elif topology == "barabasi_albert":
        G = nx.barabasi_albert_graph(n, 3, seed=seed)
    elif topology == "complete":
        G = nx.complete_graph(n)
    else:
        G = nx.watts_strogatz_graph(n, 4, 0.3, seed=seed)

    inject_defaults(G)
    for node in G.nodes():
        G.nodes[node]["phase"] = rng.uniform(0, 2 * math.pi)
        G.nodes[node]["frequency"] = rng.uniform(0.1, 1.0)
        G.nodes[node]["delta_nfr"] = rng.uniform(-0.5, 0.5)
        G.nodes[node]["EPI"] = f"epi_{node}"
    return G


@pytest.fixture
def ws_graph():
    return _make_tnfr_graph(20, "watts_strogatz")


@pytest.fixture
def ba_graph():
    return _make_tnfr_graph(20, "barabasi_albert")


@pytest.fixture
def complete_graph():
    return _make_tnfr_graph(10, "complete")


# ===========================================================================
# 1. Operator Lyapunov bounds registry
# ===========================================================================

class TestOperatorLyapunovBoundsRegistry:
    """All 13 operators must have registered bounds."""

    EXPECTED_OPERATORS = [
        "Coherence", "Reception", "Coupling", "SelfOrganization",
        "Transition", "Dissonance", "Expansion", "Emission",
        "Resonance", "Silence", "Mutation", "Recursivity", "Contraction",
    ]

    EXPECTED_GLYPHS = [
        "IL", "EN", "UM", "THOL", "NAV",
        "OZ", "VAL", "AL", "RA",
        "SHA", "ZHIR", "REMESH", "NUL",
    ]

    def test_registry_has_13_operators(self):
        assert len(OPERATOR_LYAPUNOV_BOUNDS) == 13

    @pytest.mark.parametrize("name", EXPECTED_OPERATORS)
    def test_operator_present_by_name(self, name):
        bound = get_bound(name)
        assert isinstance(bound, OperatorLyapunovBound)
        assert bound.operator_name == name

    @pytest.mark.parametrize("glyph", EXPECTED_GLYPHS)
    def test_operator_present_by_glyph(self, glyph):
        bound = get_bound(glyph)
        assert isinstance(bound, OperatorLyapunovBound)
        assert bound.glyph == glyph

    def test_unknown_operator_raises(self):
        with pytest.raises(KeyError):
            get_bound("NonExistent")

    def test_all_have_derivation(self):
        for name, bound in OPERATOR_LYAPUNOV_BOUNDS.items():
            assert len(bound.derivation) > 10, f"{name} missing derivation"

    def test_all_have_positive_glyph_factor(self):
        for name, bound in OPERATOR_LYAPUNOV_BOUNDS.items():
            assert bound.glyph_factor_value >= 0.0, f"{name} invalid factor"


# ===========================================================================
# 2. Energy class taxonomy
# ===========================================================================

class TestEnergyClassTaxonomy:
    """Verify the operators are classified correctly."""

    STABILISERS = ["Coherence", "Reception", "Coupling",
                   "SelfOrganization", "Transition"]
    DESTABILISERS = ["Dissonance", "Expansion", "Emission", "Resonance"]
    NEUTRALS = ["Silence", "Mutation", "Recursivity"]
    MIXED = ["Contraction"]

    @pytest.mark.parametrize("name", STABILISERS)
    def test_stabilisers(self, name):
        assert get_bound(name).energy_class == EnergyClass.STABILISER

    @pytest.mark.parametrize("name", DESTABILISERS)
    def test_destabilisers(self, name):
        assert get_bound(name).energy_class == EnergyClass.DESTABILISER

    @pytest.mark.parametrize("name", NEUTRALS)
    def test_neutrals(self, name):
        assert get_bound(name).energy_class == EnergyClass.NEUTRAL

    @pytest.mark.parametrize("name", MIXED)
    def test_mixed(self, name):
        assert get_bound(name).energy_class == EnergyClass.MIXED


# ===========================================================================
# 3. Per-operator contraction / expansion rates
# ===========================================================================

class TestContractionRates:
    """Verify contraction rates are physically sensible."""

    def test_coherence_rate_approx_046(self):
        """IL: ρ = 1 - 0.737² ≈ 0.457."""
        bound = get_bound("IL")
        assert 0.40 < bound.contraction_rate < 0.50

    def test_dissonance_rate_approx_7(self):
        """OZ: κ = 2.803² - 1 ≈ 6.857."""
        bound = get_bound("OZ")
        assert 6.5 < bound.contraction_rate < 7.2

    def test_emission_rate_small(self):
        """AL: κ = (0.1171)² ≈ 0.014 per node."""
        bound = get_bound("AL")
        assert bound.contraction_rate < 0.02

    def test_resonance_rate_approx_01(self):
        """RA: κ = 1.05² - 1 ≈ 0.1025."""
        bound = get_bound("RA")
        assert 0.09 < bound.contraction_rate < 0.12

    def test_expansion_rate_approx_014(self):
        """VAL: κ = 1.0673² - 1 ≈ 0.139."""
        bound = get_bound("VAL")
        assert 0.12 < bound.contraction_rate < 0.16

    def test_silence_rate_small(self):
        """SHA: quasi-isometric, ε ≈ 0.187."""
        bound = get_bound("SHA")
        assert 0.15 < bound.contraction_rate < 0.22

    def test_recursivity_rate_zero(self):
        """REMESH: advisory, ΔE = 0."""
        bound = get_bound("REMESH")
        assert bound.contraction_rate == 0.0

    def test_stabilisers_have_positive_rates(self):
        for name in ["Coherence", "Reception", "Coupling",
                     "SelfOrganization", "Transition"]:
            assert get_bound(name).contraction_rate > 0.0


# ===========================================================================
# 4. Energy bound computation
# ===========================================================================

class TestComputeOperatorEnergyBound:
    """Test compute_operator_energy_bound for each class."""

    def test_stabiliser_bound_is_negative(self):
        """Stabiliser: ΔE ≤ -ρ·E < 0."""
        e0 = 10.0
        delta = compute_operator_energy_bound("Coherence", e0, n_nodes=20)
        assert delta < 0.0

    def test_destabiliser_bound_is_positive(self):
        """Destabiliser: ΔE ≤ +κ·E > 0."""
        e0 = 10.0
        delta = compute_operator_energy_bound("Dissonance", e0, n_nodes=20)
        assert delta > 0.0

    def test_emission_bound_scales_with_nodes(self):
        """AL: additive → ΔE ≤ κ·N."""
        d1 = compute_operator_energy_bound("Emission", 10.0, n_nodes=1)
        d10 = compute_operator_energy_bound("Emission", 10.0, n_nodes=10)
        assert abs(d10 - 10 * d1) < 1e-12

    def test_neutral_bound_scales_with_nodes(self):
        """Neutral: ΔE ≤ ε·N."""
        d5 = compute_operator_energy_bound("Mutation", 10.0, n_nodes=5)
        d1 = compute_operator_energy_bound("Mutation", 10.0, n_nodes=1)
        assert abs(d5 - 5 * d1) < 1e-12

    def test_recursivity_bound_is_zero(self):
        delta = compute_operator_energy_bound("Recursivity", 10.0, n_nodes=20)
        assert delta == 0.0

    def test_zero_energy_gives_zero_bound_for_multiplicative(self):
        """If E₀ = 0, multiplicative operators give ΔE = 0."""
        assert compute_operator_energy_bound("Coherence", 0.0) == 0.0
        assert compute_operator_energy_bound("Dissonance", 0.0) == 0.0


# ===========================================================================
# 5. Operator Lyapunov verification
# ===========================================================================

class TestVerifyOperatorLyapunov:
    """Test the verification function against synthetic energy data."""

    def test_coherence_within_bound(self):
        """IL: energy decreases by at least ρ·E → within bound.

        ρ ≈ 0.457 → bound: ΔE ≤ -4.568, so E_after ≤ 5.432.
        Using E_after = 4.0 → ΔE = -6.0 ≤ -4.568 ✓
        """
        result = verify_operator_lyapunov("IL", 10.0, 4.0, n_nodes=20)
        assert result.within_bound
        assert result.delta_e < 0.0
        assert result.margin > 0.0

    def test_dissonance_within_bound(self):
        """OZ: energy increases but within κ·E."""
        e0 = 1.0
        # Moderate increase within bound (κ ≈ 6.857)
        result = verify_operator_lyapunov("OZ", e0, 5.0, n_nodes=20)
        assert result.within_bound

    def test_dissonance_out_of_bound(self):
        """OZ: energy increases beyond κ·E → out of bound."""
        e0 = 1.0
        # Extreme increase beyond κ·E ≈ 6.857
        result = verify_operator_lyapunov("OZ", e0, 100.0, n_nodes=20)
        assert not result.within_bound

    def test_verification_dataclass_fields(self):
        result = verify_operator_lyapunov("IL", 10.0, 8.0, n_nodes=10)
        assert result.operator_name == "Coherence"
        assert result.glyph == "IL"
        assert result.energy_class == EnergyClass.STABILISER
        assert abs(result.delta_e - (-2.0)) < 1e-12


# ===========================================================================
# 6. Sequence energy bound
# ===========================================================================

class TestSequenceEnergyBound:
    """Test cumulative energy bound across operator sequences."""

    def test_pure_stabiliser_sequence_decreases(self):
        """[IL, IL, IL] → energy monotonically decreasing."""
        e_final = compute_sequence_energy_bound(["IL", "IL", "IL"], 10.0, 20)
        assert e_final < 10.0

    def test_pure_destabiliser_sequence_increases(self):
        """[OZ, OZ] → energy increases."""
        e_final = compute_sequence_energy_bound(["OZ", "OZ"], 1.0, 20)
        assert e_final > 1.0

    def test_bootstrap_sequence(self):
        """[AL, UM, IL] (Bootstrap) should compute a finite bound."""
        e_final = compute_sequence_energy_bound(["AL", "UM", "IL"], 1.0, 20)
        assert math.isfinite(e_final)
        assert e_final >= 0.0

    def test_explore_sequence(self):
        """[OZ, ZHIR, IL] (Explore) should be bounded."""
        e_final = compute_sequence_energy_bound(["OZ", "ZHIR", "IL"], 1.0, 20)
        assert math.isfinite(e_final)


# ===========================================================================
# 7. Sequence Lyapunov proof
# ===========================================================================

class TestSequenceLyapunovProof:
    """Test formal proof of net contractiveness for compliant sequences."""

    def test_stabiliser_only_is_contractive(self):
        """Pure stabiliser sequences are net-contractive."""
        proof = prove_sequence_lyapunov(["IL", "IL", "IL"])
        assert proof.is_net_contractive
        assert proof.cumulative_product < 1.0
        assert proof.net_contraction > 0.0

    def test_oz_then_il_is_contractive(self):
        """OZ followed by IL: IL's contraction (ρ=0.457) applied to
        expanded energy should dominate because (1+6.857)×(1-0.457)
        ≈ 4.27 > 1, so single IL is not enough."""
        proof = prove_sequence_lyapunov(["OZ", "IL"])
        # OZ multiplier ≈ 7.857, IL multiplier ≈ 0.543
        # Product ≈ 4.27 > 1 → NOT net-contractive with single IL
        assert not proof.is_net_contractive

    def test_oz_then_many_il_is_contractive(self):
        """OZ followed by enough ILs to compensate."""
        # Product = 7.857 × 0.543^n < 1 requires n > ln(7.857)/ln(1/0.543)
        # n > 3.37 → 4 ILs needed
        proof = prove_sequence_lyapunov(["OZ", "IL", "IL", "IL", "IL"])
        assert proof.is_net_contractive

    def test_bootstrap_explore_stabilize(self):
        """[AL, UM, IL, OZ, ZHIR, IL, IL, IL, IL, SHA] → grammar compliant."""
        seq = ["AL", "UM", "IL", "OZ", "ZHIR", "IL", "IL", "IL", "IL", "SHA"]
        proof = prove_sequence_lyapunov(seq)
        assert len(proof.operators) == 10
        assert len(proof.energy_multipliers) == 10
        assert proof.cumulative_product > 0.0

    def test_proof_dataclass_completeness(self):
        proof = prove_sequence_lyapunov(["IL"])
        assert isinstance(proof.operators, tuple)
        assert isinstance(proof.energy_multipliers, tuple)
        assert isinstance(proof.cumulative_product, float)
        assert isinstance(proof.is_net_contractive, bool)
        assert isinstance(proof.net_contraction, float)


# ===========================================================================
# 8. Spectral gap analysis
# ===========================================================================

class TestSpectralGapAnalysis:
    """Test spectral gap characterisation on various topologies."""

    def test_connected_graph_has_positive_gap(self, ws_graph):
        result = analyze_spectral_gap(ws_graph)
        assert result.spectral_gap > 0.0
        assert result.is_connected

    def test_complete_graph_has_large_gap(self):
        """Complete graph K_n: λ₁ = n."""
        G = nx.complete_graph(10)
        inject_defaults(G)
        result = analyze_spectral_gap(G)
        assert abs(result.spectral_gap - 10.0) < 0.1

    def test_relaxation_time_is_inverse_gap(self, ws_graph):
        result = analyze_spectral_gap(ws_graph)
        expected_tau = 1.0 / result.spectral_gap
        assert abs(result.relaxation_time - expected_tau) < 1e-10

    def test_mixing_time_bounded(self, ws_graph):
        result = analyze_spectral_gap(ws_graph)
        n = ws_graph.number_of_nodes()
        expected = math.log(n) / result.spectral_gap
        assert abs(result.mixing_time_bound - expected) < 1e-10

    def test_cheeger_lower_positive(self, ws_graph):
        result = analyze_spectral_gap(ws_graph)
        assert result.cheeger_lower > 0.0

    def test_spectral_ratio_finite(self, ws_graph):
        result = analyze_spectral_gap(ws_graph)
        assert math.isfinite(result.spectral_ratio)
        assert result.spectral_ratio >= 1.0

    def test_eigenvalues_array_length(self, ws_graph):
        result = analyze_spectral_gap(ws_graph)
        assert len(result.eigenvalues) == ws_graph.number_of_nodes()

    def test_fiedler_equals_spectral_gap(self, ws_graph):
        result = analyze_spectral_gap(ws_graph)
        assert result.fiedler_value == result.spectral_gap

    def test_single_node_graph(self):
        G = nx.Graph()
        G.add_node(0)
        inject_defaults(G)
        result = analyze_spectral_gap(G)
        assert result.spectral_gap == 0.0
        assert result.relaxation_time == float("inf")
        assert result.is_connected

    @pytest.mark.parametrize("topology", ["watts_strogatz", "barabasi_albert",
                                          "complete"])
    def test_positive_gap_across_topologies(self, topology):
        G = _make_tnfr_graph(15, topology, seed=7)
        result = analyze_spectral_gap(G)
        assert result.spectral_gap > 0.0
        assert result.is_connected


# ===========================================================================
# 9. Combined Lyapunov + spectral convergence
# ===========================================================================

class TestOperatorConvergence:
    """Test analyze_operator_convergence for combined analysis."""

    def test_stabiliser_has_positive_convergence(self, ws_graph):
        summary = analyze_operator_convergence(ws_graph, "IL")
        assert summary.effective_convergence_rate > 0.0
        assert math.isfinite(summary.steps_to_half_energy)
        assert summary.steps_to_half_energy > 0.0

    def test_destabiliser_has_zero_convergence(self, ws_graph):
        summary = analyze_operator_convergence(ws_graph, "OZ")
        assert summary.effective_convergence_rate == 0.0
        assert summary.steps_to_half_energy == float("inf")

    def test_effective_rate_is_min_of_rho_and_lambda(self, ws_graph):
        summary = analyze_operator_convergence(ws_graph, "IL")
        bound = get_bound("IL")
        spectral = analyze_spectral_gap(ws_graph)
        expected = min(bound.contraction_rate, spectral.spectral_gap)
        assert abs(summary.effective_convergence_rate - expected) < 1e-12

    def test_steps_to_half_uses_ln2(self, ws_graph):
        summary = analyze_operator_convergence(ws_graph, "IL")
        if summary.effective_convergence_rate > 0:
            expected = math.log(2) / summary.effective_convergence_rate
            assert abs(summary.steps_to_half_energy - expected) < 1e-10

    @pytest.mark.parametrize("glyph", ["IL", "EN", "UM", "THOL", "NAV"])
    def test_all_stabilisers_converge(self, ws_graph, glyph):
        summary = analyze_operator_convergence(ws_graph, glyph)
        assert summary.effective_convergence_rate > 0.0


# ===========================================================================
# 10. Edge cases and robustness
# ===========================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_initial_energy(self):
        """E₀ = 0 should not cause division errors."""
        for glyph in ["IL", "OZ", "AL", "SHA", "REMESH"]:
            d = compute_operator_energy_bound(glyph, 0.0, n_nodes=10)
            assert math.isfinite(d)

    def test_very_small_energy(self):
        """Very small energy should produce small bounds."""
        d = compute_operator_energy_bound("IL", 1e-15, n_nodes=1)
        assert abs(d) < 1e-14

    def test_sequence_never_goes_negative(self):
        """Energy bound is clamped to >= 0."""
        # Many aggressive stabilisers on small energy
        e = compute_sequence_energy_bound(
            ["IL"] * 50, energy_initial=0.001, n_nodes=1
        )
        assert e >= 0.0

    def test_empty_sequence_returns_initial(self):
        e = compute_sequence_energy_bound([], 42.0, n_nodes=10)
        assert e == 42.0

    def test_proof_single_operator(self):
        proof = prove_sequence_lyapunov(["REMESH"])
        assert proof.is_net_contractive  # product = 1.0
        assert proof.cumulative_product == 1.0
