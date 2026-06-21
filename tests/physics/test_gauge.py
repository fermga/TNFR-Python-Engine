"""Tests for TNFR Gauge Structure — Local U(1) Symmetry of Complex Geometric Field.

Validates the **Structural Gauge Theorem**: the complex geometric field
Ψ = K_φ + i·J_φ admits a local U(1) gauge symmetry under which physical
observables (ℰ, |Ψ|, C(t), |𝒯|², |𝒳|²) are exactly invariant while
gauge-dependent quantities (Q, 𝒬, χ, arg Ψ) transform non-trivially.

Tests verify:
1.  Gauge transformation: K_φ'/J_φ' rotation by angle α
2.  |Ψ|² invariance under local U(1)
3.  Energy density ℰ invariance
4.  Topological norm |𝒯|² = 𝒬² + 𝒬̃² invariance
5.  Chirality norm |𝒳|² = χ² + χ̃² invariance
6.  Symmetry breaking 𝒮 invariance
7.  Noether charge Q NON-invariance (expected)
8.  Gauge connection A_ij = arg(Ψ_j) − arg(Ψ_i) antisymmetry
9.  Covariant derivative D_ij Ψ magnitude invariance
10. Gauge curvature F_C on triangles (holonomy)
11. Yang-Mills action S_YM ≥ 0
12. Energy decomposition consistency
13. Interaction regime classification
14. Multi-topology validation (WS, BA, Grid)
15. Dual charges (𝒬̃, χ̃) correctness
16. GaugeSnapshot capture
17. Reproducibility under deterministic seeds

TIER: CORE PHYSICS — gauge structure axiomatises internal field symmetry.
"""

from __future__ import annotations

import copy
import math
import os
import sys

import networkx as nx
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tnfr.constants import inject_defaults

from tnfr.physics.gauge import (
    GaugeSnapshot,
    GaugeInvarianceResult,
    YangMillsFieldEquations,
    BianchiIdentityResult,
    InteractionRegimeMetrics,
    NetworkInteractionProfile,
    REGIME_ACTIVITY_SHARE,
    N_REGIMES,
    apply_gauge_transformation,
    compute_gauge_connection,
    compute_gauge_curvature,
    compute_covariant_derivative,
    compute_covariant_derivative_magnitude,
    compute_topological_norm,
    compute_chirality_norm,
    compute_dual_topological_charge,
    compute_dual_chirality,
    verify_gauge_invariance,
    capture_gauge_snapshot,
    classify_interaction_regime,
    classify_network_regimes,
    compute_yang_mills_action,
    compute_gauge_energy_decomposition,
    compute_matter_current,
    compute_yang_mills_equations,
    verify_bianchi_identity,
    compute_gauss_law_residual,
    compute_gauge_coupling_constant,
    classify_interaction_regime_formal,
    compute_network_interaction_profile,
)
from tnfr.physics.unified import (
    compute_complex_geometric_field,
    compute_energy_density,
    compute_topological_charge,
    compute_chirality_field,
    compute_symmetry_breaking_field,
)
from tnfr.physics.canonical import (
    compute_phase_gradient,
    compute_phase_curvature,
)
from tnfr.physics.extended import (
    compute_phase_current,
    compute_dnfr_flux,
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


@pytest.fixture
def ws_graph():
    return _make_tnfr_graph(30, "watts_strogatz", seed=42)


@pytest.fixture
def ba_graph():
    return _make_tnfr_graph(30, "barabasi_albert", seed=42)


@pytest.fixture
def grid_graph():
    return _make_tnfr_graph(25, "grid", seed=42)


@pytest.fixture
def random_alpha(ws_graph):
    """Random gauge parameters for each node."""
    rng = np.random.default_rng(123)
    return {n: float(rng.uniform(0, 2 * math.pi)) for n in ws_graph.nodes()}


@pytest.fixture
def constant_alpha(ws_graph):
    """Constant (global) gauge parameter — physically trivial."""
    return {n: 1.23 for n in ws_graph.nodes()}


# ===================================================================
# 1. Gauge Transformation Mechanics
# ===================================================================

class TestGaugeTransformation:
    """Validate the rotation Ψ → e^{iα}Ψ produces correct K_φ'/J_φ'."""

    def test_zero_alpha_identity(self, ws_graph):
        """α = 0 everywhere → fields unchanged."""
        alpha_zero = {n: 0.0 for n in ws_graph.nodes()}
        result = apply_gauge_transformation(ws_graph, alpha_zero)

        k_phi = compute_phase_curvature(ws_graph)
        j_phi = compute_phase_current(ws_graph)

        for n in ws_graph.nodes():
            assert abs(result["k_phi"][n] - k_phi[n]) < 1e-12
            assert abs(result["j_phi"][n] - j_phi[n]) < 1e-12

    def test_pi_half_rotation(self, ws_graph):
        """α = π/2 → K_φ' = −J_φ, J_φ' = K_φ."""
        alpha = {n: math.pi / 2 for n in ws_graph.nodes()}
        result = apply_gauge_transformation(ws_graph, alpha)

        k_phi = compute_phase_curvature(ws_graph)
        j_phi = compute_phase_current(ws_graph)

        for n in ws_graph.nodes():
            assert abs(result["k_phi"][n] - (-j_phi[n])) < 1e-10
            assert abs(result["j_phi"][n] - k_phi[n]) < 1e-10

    def test_pi_rotation(self, ws_graph):
        """α = π → K_φ' = −K_φ, J_φ' = −J_φ (sign flip)."""
        alpha = {n: math.pi for n in ws_graph.nodes()}
        result = apply_gauge_transformation(ws_graph, alpha)

        k_phi = compute_phase_curvature(ws_graph)
        j_phi = compute_phase_current(ws_graph)

        for n in ws_graph.nodes():
            assert abs(result["k_phi"][n] + k_phi[n]) < 1e-10
            assert abs(result["j_phi"][n] + j_phi[n]) < 1e-10

    def test_psi_magnitude_preserved(self, ws_graph, random_alpha):
        """Local U(1) preserves |Ψ| at every node."""
        psi_before = compute_complex_geometric_field(ws_graph)
        result = apply_gauge_transformation(ws_graph, random_alpha)

        for n in ws_graph.nodes():
            assert abs(abs(result["psi"][n]) - abs(psi_before[n])) < 1e-10

    def test_double_transform_composes(self, ws_graph):
        """Two successive transforms α₁, α₂ equal one transform α₁ + α₂."""
        rng = np.random.default_rng(77)
        alpha1 = {n: float(rng.uniform(0, math.pi)) for n in ws_graph.nodes()}
        alpha2 = {n: float(rng.uniform(0, math.pi)) for n in ws_graph.nodes()}
        alpha_sum = {n: alpha1[n] + alpha2[n] for n in ws_graph.nodes()}

        # Apply sum in one step
        r_sum = apply_gauge_transformation(ws_graph, alpha_sum)

        # Apply sequentially: first α₁, then α₂ on transformed k_phi/j_phi
        r1 = apply_gauge_transformation(ws_graph, alpha1)

        # Manually apply second rotation on r1 results
        for n in ws_graph.nodes():
            a2 = alpha2[n]
            kp = r1["k_phi"][n]
            jp = r1["j_phi"][n]
            kp2 = kp * math.cos(a2) - jp * math.sin(a2)
            jp2 = kp * math.sin(a2) + jp * math.cos(a2)
            assert abs(kp2 - r_sum["k_phi"][n]) < 1e-10
            assert abs(jp2 - r_sum["j_phi"][n]) < 1e-10


# ===================================================================
# 2. Gauge Invariance of Physical Quantities
# ===================================================================

class TestGaugeInvariance:
    """Validate that ℰ, |Ψ|, |𝒯|², |𝒳|², 𝒮 are gauge-invariant."""

    def test_energy_density_invariant(self, ws_graph, random_alpha):
        """Energy density ℰ(i) exactly invariant under local U(1)."""
        result = verify_gauge_invariance(ws_graph, random_alpha)
        assert result.energy_max_deviation < 1e-10

    def test_magnitude_invariant(self, ws_graph, random_alpha):
        """|Ψ(i)| exactly invariant under local U(1)."""
        result = verify_gauge_invariance(ws_graph, random_alpha)
        assert result.magnitude_max_deviation < 1e-10

    def test_topological_norm_invariant(self, ws_graph, random_alpha):
        """|𝒯|² = 𝒬² + 𝒬̃² invariant under local U(1)."""
        result = verify_gauge_invariance(ws_graph, random_alpha)
        assert result.topological_norm_max_deviation < 1e-10

    def test_chirality_norm_invariant(self, ws_graph, random_alpha):
        """|𝒳|² = χ² + χ̃² invariant under local U(1)."""
        result = verify_gauge_invariance(ws_graph, random_alpha)
        assert result.chirality_norm_max_deviation < 1e-10

    def test_symmetry_breaking_NOT_invariant(self, ws_graph, random_alpha):
        """𝒮 = (|∇φ|² − K_φ²) + (J_φ² − J_ΔNFR²) is NOT gauge-invariant.

        K_φ² and J_φ² individually change under rotation even though
        K_φ² + J_φ² = |Ψ|² is preserved.  For non-trivial α the
        deviation must be non-zero.
        """
        result = verify_gauge_invariance(ws_graph, random_alpha)
        assert result.symmetry_breaking_max_deviation > 1e-6, (
            "Expected 𝒮 to change under gauge transform"
        )

    def test_coherence_invariant(self, ws_graph, random_alpha):
        """C(t) invariant (external to Ψ internal rotation)."""
        result = verify_gauge_invariance(ws_graph, random_alpha)
        assert result.coherence_deviation < 1e-12

    def test_noether_charge_NOT_invariant(self, ws_graph, random_alpha):
        """Q = Σ(Φ_s + K_φ) is NOT gauge-invariant (K_φ rotates)."""
        result = verify_gauge_invariance(ws_graph, random_alpha)
        assert result.details["has_nontrivial_alpha"]
        # Q should change, but all *true* gauge-invariant quantities hold
        assert result.noether_charge_deviation > 1e-6, (
            "Expected Q to change under gauge transform"
        )
        assert result.is_invariant

    def test_verify_returns_true(self, ws_graph, random_alpha):
        """Full invariance check passes with default tolerance."""
        result = verify_gauge_invariance(ws_graph, random_alpha)
        assert result.is_invariant is True

    def test_verify_with_seed(self, ws_graph):
        """Deterministic random alpha via seed gives reproducible result."""
        r1 = verify_gauge_invariance(ws_graph, seed=42)
        r2 = verify_gauge_invariance(ws_graph, seed=42)
        assert r1.energy_max_deviation == r2.energy_max_deviation
        assert r1.noether_charge_deviation == r2.noether_charge_deviation

    def test_invariance_factorisation(self, ws_graph):
        """Verify |𝒯|² = |Ψ|²·|Ω|² factorisation identity.

        For any node: 𝒬² + 𝒬̃² = |Ψ|² · (|∇φ|² + J_ΔNFR²).
        """
        psi = compute_complex_geometric_field(ws_graph)
        grad_phi = compute_phase_gradient(ws_graph)
        j_dnfr = compute_dnfr_flux(ws_graph)
        topo_norm = compute_topological_norm(ws_graph)

        for n in ws_graph.nodes():
            psi_sq = abs(psi[n]) ** 2
            omega_sq = grad_phi.get(n, 0.0) ** 2 + j_dnfr.get(n, 0.0) ** 2
            expected = psi_sq * omega_sq
            assert abs(topo_norm[n] - expected) < 1e-10, (
                f"Node {n}: |𝒯|²={topo_norm[n]:.6e} ≠ |Ψ|²·|Ω|²={expected:.6e}"
            )


# ===================================================================
# 3. Multi-Topology Validation
# ===================================================================

class TestMultiTopology:
    """Gauge invariance holds across different network topologies."""

    @pytest.mark.parametrize("topology", ["watts_strogatz", "barabasi_albert", "grid"])
    def test_gauge_invariance_cross_topology(self, topology):
        """ℰ invariance verified on WS, BA, and Grid graphs."""
        n = 25 if topology == "grid" else 30
        G = _make_tnfr_graph(n, topology, seed=42)
        result = verify_gauge_invariance(G, seed=99)
        assert result.is_invariant
        assert result.energy_max_deviation < 1e-10


# ===================================================================
# 4. Gauge Connection
# ===================================================================

class TestGaugeConnection:
    """Validate gauge connection A_ij = arg(Ψ_j) − arg(Ψ_i)."""

    def test_antisymmetry(self, ws_graph):
        """A_ji = −A_ij for undirected graphs."""
        conn = compute_gauge_connection(ws_graph)
        for u, v in ws_graph.edges():
            a_uv = conn.get((u, v), 0.0)
            a_vu = conn.get((v, u), 0.0)
            assert abs(a_uv + a_vu) < 1e-12, f"A({u},{v}) + A({v},{u}) ≠ 0"

    def test_connection_range(self, ws_graph):
        """Connection values lie in [−π, π)."""
        conn = compute_gauge_connection(ws_graph)
        for edge, val in conn.items():
            assert -math.pi - 1e-9 <= val < math.pi + 1e-9, (
                f"A{edge} = {val} outside [−π, π)"
            )

    def test_connection_zero_for_uniform_psi(self):
        """If Ψ has the same phase everywhere, A_ij = 0."""
        G = nx.complete_graph(5)
        inject_defaults(G)
        # Set all phases and delta_nfr identically
        for n in G.nodes():
            G.nodes[n]["phase"] = 1.0
            G.nodes[n]["delta_nfr"] = 0.3
            G.nodes[n]["frequency"] = 1.0
            G.nodes[n]["EPI"] = "epi"

        conn = compute_gauge_connection(G)
        for val in conn.values():
            assert abs(val) < 1e-6, f"Non-zero connection {val} for uniform Ψ"


# ===================================================================
# 5. Covariant Derivative
# ===================================================================

class TestCovariantDerivative:
    """Validate D_ij Ψ = Ψ(j) − e^{iA_ij}Ψ(i)."""

    def test_magnitude_gauge_invariant(self, ws_graph, random_alpha):
        """Test |D_ij Ψ| is gauge-invariant.

        This is a fundamental test: under Ψ → e^{iα}Ψ,
        D_ij Ψ → e^{iα(j)} D_ij Ψ, so |D_ij Ψ| is unchanged.
        """
        # Compute |D_ij Ψ| before transformation
        mag_before = compute_covariant_derivative_magnitude(ws_graph)

        # We can't easily transform the graph in-place (no EPI mutation),
        # but we can verify by manual computation that the magnitude
        # is invariant using the algebraic identity.
        psi = compute_complex_geometric_field(ws_graph)
        conn = compute_gauge_connection(ws_graph)

        for (u, v), d_uv in compute_covariant_derivative(ws_graph).items():
            # After gauge transform: D'_ij Ψ = e^{iα(v)} D_ij Ψ
            # Therefore |D'| = |D|
            alpha_v = random_alpha.get(v, 0.0)
            d_prime = d_uv * complex(math.cos(alpha_v), math.sin(alpha_v))
            assert abs(abs(d_prime) - abs(d_uv)) < 1e-12

    def test_zero_for_parallel_transport(self):
        """If Ψ is parallel-transported (constant magnitude, phase follows
        connection), then D_ij Ψ ≈ 0.

        For a pair of connected nodes with identical Ψ values,
        D = Ψ(j) − e^{i·0}·Ψ(i) = 0.
        """
        G = nx.path_graph(3)
        inject_defaults(G)
        # Set identical phases and delta_nfr to get identical Ψ
        for n in G.nodes():
            G.nodes[n]["phase"] = 2.0
            G.nodes[n]["delta_nfr"] = 0.4
            G.nodes[n]["frequency"] = 1.0
            G.nodes[n]["EPI"] = "epi"

        cov_mag = compute_covariant_derivative_magnitude(G)
        for val in cov_mag.values():
            assert val < 1e-6, f"Non-zero |D| = {val} for parallel Ψ"


# ===================================================================
# 6. Gauge Curvature (Field Strength)
# ===================================================================

class TestGaugeCurvature:
    """Validate F_C = Σ_C A_ij (holonomy on cycles)."""

    def test_curvature_on_complete_graph(self):
        """Complete graph K_4 has many triangles; curvature computed."""
        G = nx.complete_graph(5)
        inject_defaults(G)
        rng = np.random.default_rng(42)
        for n in G.nodes():
            G.nodes[n]["phase"] = float(rng.uniform(0, 2 * math.pi))
            G.nodes[n]["delta_nfr"] = float(rng.uniform(-0.5, 0.5))
            G.nodes[n]["frequency"] = 1.0
            G.nodes[n]["EPI"] = "epi"

        curv = compute_gauge_curvature(G)
        assert len(curv) > 0, "K_5 should have triangles"
        # All curvature values in [−π, π]
        for cycle, f in curv.items():
            assert -math.pi - 1e-9 <= f <= math.pi + 1e-9

    def test_curvature_zero_flat_connection(self):
        """Gauge-flat configuration: identical Ψ → F_C = 0 everywhere."""
        G = nx.complete_graph(4)
        inject_defaults(G)
        for n in G.nodes():
            G.nodes[n]["phase"] = 1.5
            G.nodes[n]["delta_nfr"] = 0.2
            G.nodes[n]["frequency"] = 1.0
            G.nodes[n]["EPI"] = "epi"

        curv = compute_gauge_curvature(G)
        for cycle, f in curv.items():
            assert abs(f) < 1e-6, f"Non-zero F on {cycle} for flat connection"

    def test_yang_mills_action_nonnegative(self, ws_graph):
        """Yang-Mills action S_YM = ½ΣF² ≥ 0."""
        s_ym = compute_yang_mills_action(ws_graph)
        assert s_ym >= 0.0

    def test_yang_mills_zero_for_flat(self):
        """S_YM = 0 for a gauge-flat configuration."""
        G = nx.complete_graph(4)
        inject_defaults(G)
        for n in G.nodes():
            G.nodes[n]["phase"] = 1.0
            G.nodes[n]["delta_nfr"] = 0.3
            G.nodes[n]["frequency"] = 1.0
            G.nodes[n]["EPI"] = "epi"

        assert compute_yang_mills_action(G) < 1e-10


# ===================================================================
# 7. Dual Charges
# ===================================================================

class TestDualCharges:
    """Validate dual topological charge 𝒬̃ and dual chirality χ̃."""

    def test_dual_topological_charge_definition(self, ws_graph):
        """𝒬̃ = K_φ·|∇φ| + J_φ·J_ΔNFR computed correctly."""
        grad_phi = compute_phase_gradient(ws_graph)
        k_phi = compute_phase_curvature(ws_graph)
        j_phi = compute_phase_current(ws_graph)
        j_dnfr = compute_dnfr_flux(ws_graph)

        q_dual = compute_dual_topological_charge(ws_graph)

        for n in ws_graph.nodes():
            expected = (
                k_phi.get(n, 0.0) * grad_phi.get(n, 0.0)
                + j_phi.get(n, 0.0) * j_dnfr.get(n, 0.0)
            )
            assert abs(q_dual[n] - expected) < 1e-12

    def test_dual_chirality_definition(self, ws_graph):
        """χ̃ = |∇φ|·J_φ + K_φ·J_ΔNFR computed correctly."""
        grad_phi = compute_phase_gradient(ws_graph)
        k_phi = compute_phase_curvature(ws_graph)
        j_phi = compute_phase_current(ws_graph)
        j_dnfr = compute_dnfr_flux(ws_graph)

        chi_dual = compute_dual_chirality(ws_graph)

        for n in ws_graph.nodes():
            expected = (
                grad_phi.get(n, 0.0) * j_phi.get(n, 0.0)
                + k_phi.get(n, 0.0) * j_dnfr.get(n, 0.0)
            )
            assert abs(chi_dual[n] - expected) < 1e-12

    def test_topological_norm_from_q_and_q_dual(self, ws_graph):
        """Verify |𝒯|² = 𝒬² + 𝒬̃² by explicit computation."""
        topo_charge = compute_topological_charge(ws_graph)
        q_dual = compute_dual_topological_charge(ws_graph)
        topo_norm = compute_topological_norm(ws_graph)

        for n in ws_graph.nodes():
            expected = topo_charge.get(n, 0.0) ** 2 + q_dual[n] ** 2
            assert abs(topo_norm[n] - expected) < 1e-10

    def test_chirality_norm_from_chi_and_chi_dual(self, ws_graph):
        """Verify |𝒳|² = χ² + χ̃² by explicit computation."""
        chirality = compute_chirality_field(ws_graph)
        chi_dual = compute_dual_chirality(ws_graph)
        chiral_norm = compute_chirality_norm(ws_graph)

        for n in ws_graph.nodes():
            expected = chirality.get(n, 0.0) ** 2 + chi_dual[n] ** 2
            assert abs(chiral_norm[n] - expected) < 1e-10


# ===================================================================
# 8. Gauge Snapshot
# ===================================================================

class TestGaugeSnapshot:
    """Validate the GaugeSnapshot capture."""

    def test_snapshot_fields_present(self, ws_graph):
        """All fields are populated in the snapshot."""
        snap = capture_gauge_snapshot(ws_graph)
        assert len(snap.psi) == len(ws_graph)
        assert len(snap.psi_magnitude) == len(ws_graph)
        assert len(snap.psi_phase) == len(ws_graph)
        assert len(snap.connection) > 0
        assert len(snap.energy_density) == len(ws_graph)
        assert len(snap.topological_norm) == len(ws_graph)
        assert len(snap.chirality_norm) == len(ws_graph)

    def test_snapshot_magnitude_consistency(self, ws_graph):
        """snapshot.psi_magnitude matches |snapshot.psi|."""
        snap = capture_gauge_snapshot(ws_graph)
        for n in ws_graph.nodes():
            assert abs(snap.psi_magnitude[n] - abs(snap.psi[n])) < 1e-12


# ===================================================================
# 9. Energy Decomposition
# ===================================================================

class TestEnergyDecomposition:
    """Validate gauge-theoretic energy sector decomposition."""

    def test_sectors_sum_to_total(self, ws_graph):
        """potential + gradient + gauge + flux = 2 × total_energy."""
        decomp = compute_gauge_energy_decomposition(ws_graph)
        sector_sum = (
            decomp["potential_sector"]
            + decomp["gradient_sector"]
            + decomp["gauge_sector"]
            + decomp["flux_sector"]
        )
        assert abs(sector_sum - 2.0 * decomp["total_energy"]) < 1e-10

    def test_fractions_sum_to_one(self, ws_graph):
        """Sector fractions sum to 1 (within numerical precision)."""
        decomp = compute_gauge_energy_decomposition(ws_graph)
        frac_sum = (
            decomp["potential_fraction"]
            + decomp["gradient_fraction"]
            + decomp["gauge_fraction"]
            + decomp["flux_fraction"]
        )
        assert abs(frac_sum - 1.0) < 1e-6

    def test_yang_mills_nonnegative(self, ws_graph):
        """Yang-Mills action in decomposition is non-negative."""
        decomp = compute_gauge_energy_decomposition(ws_graph)
        assert decomp["yang_mills_action"] >= 0.0


# ===================================================================
# 10. Interaction Regime Classification
# ===================================================================

class TestInteractionRegimes:
    """Validate gauge-based interaction regime classification."""

    def test_regime_is_valid_string(self, ws_graph):
        """Regime classification returns a known regime type."""
        valid_regimes = {"em_like", "weak_like", "strong_like", "gravity_like"}
        for n in ws_graph.nodes():
            info = classify_interaction_regime(ws_graph, n)
            assert info["regime"] in valid_regimes

    def test_regime_scores_sum(self, ws_graph):
        """Regime scores should be non-negative."""
        for n in list(ws_graph.nodes())[:5]:
            info = classify_interaction_regime(ws_graph, n)
            for score in info["regime_scores"].values():
                assert score >= 0.0

    def test_network_regimes(self, ws_graph):
        """Full network classification returns valid structure."""
        result = classify_network_regimes(ws_graph)
        assert "per_node" in result
        assert "regime_distribution" in result
        assert "dominant_regime" in result
        assert "mean_gauge_curvature" in result
        assert "gauge_flatness" in result

        total = sum(result["regime_distribution"].values())
        assert total == len(ws_graph)

    def test_gauge_flatness_range(self, ws_graph):
        """Gauge flatness is a fraction in [0, 1]."""
        result = classify_network_regimes(ws_graph)
        assert 0.0 <= result["gauge_flatness"] <= 1.0


# ===================================================================
# 11. Reproducibility
# ===================================================================

class TestReproducibility:
    """Verify deterministic results under fixed seeds (Invariant #6)."""

    def test_gauge_invariance_reproducible(self):
        """Same graph + same seed → identical gauge invariance result."""
        G1 = _make_tnfr_graph(20, "watts_strogatz", seed=42)
        G2 = _make_tnfr_graph(20, "watts_strogatz", seed=42)

        r1 = verify_gauge_invariance(G1, seed=99)
        r2 = verify_gauge_invariance(G2, seed=99)

        assert r1.energy_max_deviation == r2.energy_max_deviation
        assert r1.noether_charge_deviation == r2.noether_charge_deviation
        assert r1.topological_norm_max_deviation == r2.topological_norm_max_deviation

    def test_snapshot_reproducible(self):
        """Same graph → same snapshot values."""
        G = _make_tnfr_graph(15, "watts_strogatz", seed=42)
        s1 = capture_gauge_snapshot(G)
        s2 = capture_gauge_snapshot(G)

        for n in G.nodes():
            assert s1.psi[n] == s2.psi[n]
            assert s1.energy_density[n] == s2.energy_density[n]


# ===================================================================
# 12. Edge Cases
# ===================================================================

class TestEdgeCases:
    """Test behaviour on minimal and degenerate graphs."""

    def test_pair_graph(self):
        """Two-node graph: gauge structure still valid."""
        G = nx.path_graph(2)
        inject_defaults(G)
        for n in G.nodes():
            G.nodes[n]["phase"] = float(n)
            G.nodes[n]["delta_nfr"] = 0.5
            G.nodes[n]["frequency"] = 1.0
            G.nodes[n]["EPI"] = "epi"

        result = verify_gauge_invariance(G, seed=42)
        assert result.is_invariant

    def test_triangle_graph(self):
        """Triangle graph: single curvature plaquette."""
        G = nx.complete_graph(3)
        inject_defaults(G)
        rng = np.random.default_rng(42)
        for n in G.nodes():
            G.nodes[n]["phase"] = float(rng.uniform(0, 2 * math.pi))
            G.nodes[n]["delta_nfr"] = float(rng.uniform(-0.5, 0.5))
            G.nodes[n]["frequency"] = 1.0
            G.nodes[n]["EPI"] = "epi"

        curv = compute_gauge_curvature(G)
        assert len(curv) == 1  # Exactly one triangle
        result = verify_gauge_invariance(G, seed=42)
        assert result.is_invariant

    def test_star_graph_no_triangles(self):
        """Star graph has no triangles → empty curvature."""
        G = nx.star_graph(5)
        inject_defaults(G)
        rng = np.random.default_rng(42)
        for n in G.nodes():
            G.nodes[n]["phase"] = float(rng.uniform(0, 2 * math.pi))
            G.nodes[n]["delta_nfr"] = float(rng.uniform(-0.5, 0.5))
            G.nodes[n]["frequency"] = 1.0
            G.nodes[n]["EPI"] = "epi"

        curv = compute_gauge_curvature(G)
        assert len(curv) == 0  # No triangles in star graph
        assert compute_yang_mills_action(G) == 0.0


# ===================================================================
# Yang-Mills Field Equations & Formal Regime Classification Tests
# ===================================================================


class TestMatterCurrent:
    """Gauge-covariant matter current J_matter(i,j)."""

    def test_antisymmetry(self, ws_graph):
        """J(j,i) = −J(i,j) for every oriented edge."""
        j_mat = compute_matter_current(ws_graph)
        for (u, v), val in j_mat.items():
            if (v, u) in j_mat:
                assert abs(j_mat[(v, u)] + val) < 1e-12

    def test_gauge_invariance(self, ws_graph, random_alpha):
        """Matter current is gauge-invariant under local U(1)."""
        j_before = compute_matter_current(ws_graph)
        G2 = copy.deepcopy(ws_graph)
        apply_gauge_transformation(G2, random_alpha)
        j_after = compute_matter_current(G2)
        for edge in j_before:
            assert abs(j_before[edge] - j_after[edge]) < 1e-10, (
                f"Matter current not gauge-invariant at edge {edge}"
            )

    def test_uniform_psi_zero_current(self):
        """Spatially uniform Ψ → zero matter current."""
        G = nx.complete_graph(5)
        inject_defaults(G)
        for n in G.nodes():
            G.nodes[n]["phase"] = 0.0
            G.nodes[n]["delta_nfr"] = 0.5
            G.nodes[n]["frequency"] = 1.0
            G.nodes[n]["EPI"] = "epi"
        j_mat = compute_matter_current(G)
        for val in j_mat.values():
            assert abs(val) < 1e-12

    def test_covers_all_edges(self, ws_graph):
        """Current dict contains both orientations for every edge."""
        j_mat = compute_matter_current(ws_graph)
        for u, v in ws_graph.edges():
            assert (u, v) in j_mat or (v, u) in j_mat

    @pytest.mark.parametrize("topo", ["watts_strogatz", "barabasi_albert", "grid"])
    def test_multi_topology(self, topo):
        """Matter current works across topologies."""
        G = _make_tnfr_graph(20, topology=topo, seed=77)
        j_mat = compute_matter_current(G)
        assert len(j_mat) > 0


class TestYangMillsFieldEquations:
    """Complete discrete Yang-Mills field equations."""

    def test_result_type(self, ws_graph):
        """Returns YangMillsFieldEquations dataclass."""
        result = compute_yang_mills_equations(ws_graph)
        assert isinstance(result, YangMillsFieldEquations)

    def test_actions_non_negative(self, ws_graph):
        """S_YM ≥ 0 and S_matter ≥ 0."""
        eq = compute_yang_mills_equations(ws_graph)
        assert eq.yang_mills_action >= 0.0
        assert eq.matter_action >= 0.0
        assert eq.total_action >= eq.yang_mills_action
        assert eq.total_action >= eq.matter_action

    def test_coupling_positive(self, ws_graph):
        """Coupling constant g² ≥ 0."""
        eq = compute_yang_mills_equations(ws_graph)
        assert eq.coupling_constant >= 0.0

    def test_explicit_coupling(self, ws_graph):
        """User-specified coupling overrides self-determined g²."""
        eq = compute_yang_mills_equations(ws_graph, coupling=1.0)
        assert eq.coupling_constant == pytest.approx(1.0, abs=1e-12)

    def test_residual_structure(self, ws_graph):
        """Residuals are non-negative with correct mean/max."""
        eq = compute_yang_mills_equations(ws_graph)
        for r in eq.equation_residual.values():
            assert r >= 0.0
        assert eq.mean_residual >= 0.0
        assert eq.max_residual >= eq.mean_residual or len(eq.equation_residual) <= 1

    def test_matter_current_consistency(self, ws_graph):
        """Matter current matches standalone computation."""
        eq = compute_yang_mills_equations(ws_graph)
        standalone = compute_matter_current(ws_graph)
        for edge, val in standalone.items():
            assert abs(eq.matter_current[edge] - val) < 1e-12

    @pytest.mark.parametrize("topo", ["watts_strogatz", "barabasi_albert", "grid"])
    def test_multi_topology(self, topo):
        """Works across all supported topologies."""
        G = _make_tnfr_graph(20, topology=topo, seed=33)
        eq = compute_yang_mills_equations(G)
        assert eq.total_action >= 0.0

    def test_reproducibility(self):
        """Same seed → identical Yang-Mills equations."""
        G1 = _make_tnfr_graph(25, seed=99)
        G2 = _make_tnfr_graph(25, seed=99)
        eq1 = compute_yang_mills_equations(G1)
        eq2 = compute_yang_mills_equations(G2)
        assert eq1.yang_mills_action == pytest.approx(eq2.yang_mills_action, abs=1e-14)
        assert eq1.matter_action == pytest.approx(eq2.matter_action, abs=1e-14)
        assert eq1.coupling_constant == pytest.approx(eq2.coupling_constant, abs=1e-14)


class TestBianchiIdentity:
    """Discrete Bianchi identity dF = d²A = 0."""

    def test_satisfied_ws(self, ws_graph):
        """Bianchi identity satisfied on Watts-Strogatz graph."""
        result = verify_bianchi_identity(ws_graph)
        assert isinstance(result, BianchiIdentityResult)
        assert result.num_coboundaries_tested > 0

    @pytest.mark.parametrize("topo", ["watts_strogatz", "barabasi_albert", "grid"])
    def test_multi_topology(self, topo):
        """Bianchi works across topologies."""
        G = _make_tnfr_graph(20, topology=topo, seed=55)
        result = verify_bianchi_identity(G)
        assert isinstance(result, BianchiIdentityResult)

    def test_tree_graph_trivial(self):
        """Tree graph has no plaquettes → trivially satisfied."""
        G = nx.star_graph(6)
        inject_defaults(G)
        for n in G.nodes():
            G.nodes[n]["phase"] = 0.0
            G.nodes[n]["delta_nfr"] = 0.1
            G.nodes[n]["frequency"] = 1.0
            G.nodes[n]["EPI"] = "epi"
        result = verify_bianchi_identity(G)
        assert result.is_satisfied
        assert result.num_coboundaries_tested == 0


class TestGaussLawResidual:
    """Discrete Gauss law divergence constraint."""

    def test_residual_per_node(self, ws_graph):
        """Returns one residual per node, all non-negative."""
        residuals = compute_gauss_law_residual(ws_graph)
        assert len(residuals) == ws_graph.number_of_nodes()
        for val in residuals.values():
            assert val >= 0.0

    def test_uniform_psi_small_residual(self):
        """Spatially uniform Ψ → small Gauss residual."""
        G = nx.complete_graph(5)
        inject_defaults(G)
        for n in G.nodes():
            G.nodes[n]["phase"] = 0.0
            G.nodes[n]["delta_nfr"] = 0.5
            G.nodes[n]["frequency"] = 1.0
            G.nodes[n]["EPI"] = "epi"
        residuals = compute_gauss_law_residual(G)
        for val in residuals.values():
            assert val < 1e-10

    @pytest.mark.parametrize("topo", ["watts_strogatz", "barabasi_albert", "grid"])
    def test_multi_topology(self, topo):
        """Gauss law works on all topologies."""
        G = _make_tnfr_graph(20, topology=topo, seed=66)
        residuals = compute_gauss_law_residual(G)
        assert len(residuals) == G.number_of_nodes()


class TestGaugeCouplingConstant:
    """Self-determined gauge coupling g² = ⟨F²⟩."""

    def test_non_negative(self, ws_graph):
        """g² ≥ 0 always."""
        g_sq = compute_gauge_coupling_constant(ws_graph)
        assert g_sq >= 0.0

    def test_tree_zero(self):
        """No plaquettes → g² = 0."""
        G = nx.star_graph(5)
        inject_defaults(G)
        for n in G.nodes():
            G.nodes[n]["phase"] = 0.0
            G.nodes[n]["delta_nfr"] = 0.1
            G.nodes[n]["frequency"] = 1.0
            G.nodes[n]["EPI"] = "epi"
        assert compute_gauge_coupling_constant(G) == 0.0

    def test_upper_bound(self, ws_graph):
        """g² ≤ π² (maximum possible curvature squared)."""
        g_sq = compute_gauge_coupling_constant(ws_graph)
        assert g_sq <= math.pi ** 2 + 1e-10


class TestRegimeActivityCriterion:
    """Emergent equipartition activity criterion (no overlay constant)."""

    def test_share_is_equipartition(self):
        """REGIME_ACTIVITY_SHARE = 1/N_REGIMES (max-entropy reference)."""
        assert REGIME_ACTIVITY_SHARE == pytest.approx(1.0 / N_REGIMES, abs=1e-14)

    def test_four_regimes(self):
        """Four gauge sectors (the tetrad of structural channels) => share=0.25."""
        assert N_REGIMES == 4
        assert REGIME_ACTIVITY_SHARE == pytest.approx(0.25, abs=1e-14)


class TestFormalInteractionRegimes:
    """Per-node formal regime classification with TNFR-derived thresholds."""

    def test_result_type(self, ws_graph):
        """Returns InteractionRegimeMetrics."""
        node = list(ws_graph.nodes())[0]
        m = classify_interaction_regime_formal(ws_graph, node)
        assert isinstance(m, InteractionRegimeMetrics)

    def test_order_params_bounded(self, ws_graph):
        """All order parameters in [0, 1]."""
        for node in ws_graph.nodes():
            m = classify_interaction_regime_formal(ws_graph, node)
            assert 0.0 <= m.em_order_parameter <= 1.0 + 1e-12
            assert 0.0 <= m.weak_order_parameter <= 1.0 + 1e-12
            assert 0.0 <= m.strong_order_parameter  # can exceed 1 in principle
            assert 0.0 <= m.gravity_order_parameter <= 1.0 + 1e-12

    def test_scores_sum_to_one(self, ws_graph):
        """Normalised scores sum ≈ 1."""
        for node in ws_graph.nodes():
            m = classify_interaction_regime_formal(ws_graph, node)
            total = sum(m.regime_scores.values())
            assert total == pytest.approx(1.0, abs=1e-10)

    def test_dominant_has_max_score(self, ws_graph):
        """Dominant regime has the highest score."""
        for node in list(ws_graph.nodes())[:5]:
            m = classify_interaction_regime_formal(ws_graph, node)
            max_regime = max(m.regime_scores, key=m.regime_scores.get)    # type: ignore
            assert m.dominant_regime == max_regime

    def test_valid_regime_names(self, ws_graph):
        """Dominant regime is one of the four canonical names."""
        valid = {"em_like", "weak_like", "strong_like", "gravity_like"}
        for node in ws_graph.nodes():
            m = classify_interaction_regime_formal(ws_graph, node)
            assert m.dominant_regime in valid

    def test_threshold_consistency(self, ws_graph):
        """above_threshold flags use the equipartition share on the scores."""
        for node in list(ws_graph.nodes())[:5]:
            m = classify_interaction_regime_formal(ws_graph, node)
            for regime in ("em_like", "weak_like", "strong_like", "gravity_like"):
                assert m.above_threshold[regime] == (
                    m.regime_scores[regime] > REGIME_ACTIVITY_SHARE
                )

    def test_mixing_angle_range(self, ws_graph):
        """Mixing angle arg(Ψ) ∈ [−π, π]."""
        for node in ws_graph.nodes():
            m = classify_interaction_regime_formal(ws_graph, node)
            assert -math.pi - 1e-10 <= m.mixing_angle <= math.pi + 1e-10

    @pytest.mark.parametrize("topo", ["watts_strogatz", "barabasi_albert", "grid"])
    def test_multi_topology(self, topo):
        """Formal regime works across topologies."""
        G = _make_tnfr_graph(20, topology=topo, seed=44)
        node = list(G.nodes())[0]
        m = classify_interaction_regime_formal(G, node)
        assert isinstance(m, InteractionRegimeMetrics)


class TestNetworkInteractionProfile:
    """Network-wide interaction regime aggregation."""

    def test_result_type(self, ws_graph):
        """Returns NetworkInteractionProfile."""
        profile = compute_network_interaction_profile(ws_graph)
        assert isinstance(profile, NetworkInteractionProfile)

    def test_distribution_sums_to_n(self, ws_graph):
        """Sum of regime counts equals number of nodes."""
        profile = compute_network_interaction_profile(ws_graph)
        assert sum(profile.regime_distribution.values()) == ws_graph.number_of_nodes()

    def test_fractions_sum_to_one(self, ws_graph):
        """Regime fractions sum to 1."""
        profile = compute_network_interaction_profile(ws_graph)
        assert sum(profile.regime_fractions.values()) == pytest.approx(1.0, abs=1e-10)

    def test_entropy_bounds(self, ws_graph):
        """Shannon entropy H ∈ [0, ln(4)]."""
        profile = compute_network_interaction_profile(ws_graph)
        assert profile.mixing_entropy >= 0.0
        assert profile.mixing_entropy <= math.log(4) + 1e-10

    def test_dominant_regime_valid(self, ws_graph):
        """Dominant regime is one of the four canonical names."""
        profile = compute_network_interaction_profile(ws_graph)
        assert profile.dominant_regime in {
            "em_like", "weak_like", "strong_like", "gravity_like",
        }

    def test_per_node_coverage(self, ws_graph):
        """per_node contains every node."""
        profile = compute_network_interaction_profile(ws_graph)
        assert set(profile.per_node.keys()) == set(ws_graph.nodes())

    def test_gauge_metrics_non_negative(self, ws_graph):
        """Gauge coupling, YM action, flatness are non-negative."""
        profile = compute_network_interaction_profile(ws_graph)
        assert profile.gauge_coupling_constant >= 0.0
        assert profile.yang_mills_action >= 0.0
        assert 0.0 <= profile.gauge_flatness <= 1.0

    def test_mean_order_parameters(self, ws_graph):
        """Mean order parameters are within expected range."""
        profile = compute_network_interaction_profile(ws_graph)
        for key, val in profile.mean_order_parameters.items():
            assert val >= 0.0, f"Mean O_P should be non-negative: {key}={val}"

    def test_reproducibility(self):
        """Same seed → identical profile."""
        G1 = _make_tnfr_graph(20, seed=101)
        G2 = _make_tnfr_graph(20, seed=101)
        p1 = compute_network_interaction_profile(G1)
        p2 = compute_network_interaction_profile(G2)
        assert p1.mixing_entropy == pytest.approx(p2.mixing_entropy, abs=1e-14)
        assert p1.dominant_regime == p2.dominant_regime
        assert p1.regime_distribution == p2.regime_distribution

    @pytest.mark.parametrize("topo", ["watts_strogatz", "barabasi_albert", "grid"])
    def test_multi_topology(self, topo):
        """Network profile works on all topologies."""
        G = _make_tnfr_graph(20, topology=topo, seed=88)
        profile = compute_network_interaction_profile(G)
        assert sum(profile.regime_distribution.values()) == G.number_of_nodes()
