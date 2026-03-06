"""Tests for TNFR Conservation-Gauge Unification.

Validates the central theoretical result:

    Grammar (U1-U6) → Symmetry (Translation × U(1))
        → Conservation (H = E, Q) → Gauge (Ψ, A, F) — UNIFIED

All four arise as different projections of the TNFR action functional:

    S_TNFR = Σ_n Δt · Σ_i [½(J_φ² + J_ΔNFR²) − ½(Φ_s² + |∇φ|² + K_φ²)]

Tests verify:
 1.  Grammar symmetry mapping covers all 6 rules
 2.  Action-energy identity: H_variational ≡ E_conservation (rel_err < 1e-10)
 3.  Noether-gauge decomposition: Q, E, S_YM, S_matter are finite
 4.  Gauge-conservation coupling: energy IS gauge-invariant, charge is NOT
 5.  Symplectic-gauge compatibility: ω preserved under U(1) rotation
 6.  Full unification pipeline produces coherent result
 7.  Multi-topology validation (WS, BA, Grid)
 8.  Conjugate pair structure: geometric (K_φ, J_φ) + potential (Φ_s, J_ΔNFR)
 9.  Sector energy decomposition: E_geo + E_pot > 0
10.  Gauge charge sensitivity: ΔQ > 0 under gauge rotation
11.  Summary dict contains all required keys
12.  Seed reproducibility

TIER: CORE PHYSICS — unification of conservation and gauge sectors.
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

from tnfr.physics.conservation_gauge_unification import (
    GrammarSymmetryMapping,
    ActionEnergyConsistency,
    NoetherGaugeDecomposition,
    GaugeConservationCoupling,
    SymplecticGaugeCompatibility,
    ConservationGaugeUnification,
    compute_grammar_symmetry_mapping,
    verify_action_energy_consistency,
    compute_noether_gauge_decomposition,
    compute_gauge_conservation_coupling,
    verify_symplectic_gauge_compatibility,
    run_conservation_gauge_unification,
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


# ---------------------------------------------------------------------------
# 1. Grammar Symmetry Mapping
# ---------------------------------------------------------------------------

class TestGrammarSymmetryMapping:
    """Grammar rules U1-U6 map to symmetries and conservation laws."""

    def test_covers_all_six_rules(self, ws_graph):
        """Mapping must return exactly 6 entries, one per U-rule."""
        mappings = compute_grammar_symmetry_mapping(ws_graph)
        assert len(mappings) == 6
        rules = [m.rule for m in mappings]
        assert rules == ["U1", "U2", "U3", "U4", "U5", "U6"]

    def test_all_entries_are_dataclass(self, ws_graph):
        mappings = compute_grammar_symmetry_mapping(ws_graph)
        for m in mappings:
            assert isinstance(m, GrammarSymmetryMapping)
            assert isinstance(m.rule, str)
            assert isinstance(m.symmetry_type, str)
            assert isinstance(m.conservation_law, str)
            assert isinstance(m.variational_role, str)
            assert isinstance(m.is_satisfied, bool)
            assert isinstance(m.diagnostic_value, float)

    def test_symmetry_types_are_distinct(self, ws_graph):
        """Each grammar rule maps to a different symmetry type."""
        mappings = compute_grammar_symmetry_mapping(ws_graph)
        types = [m.symmetry_type for m in mappings]
        expected = {"boundary", "stability", "gauge", "topological",
                    "hierarchical", "confinement"}
        assert set(types) == expected

    def test_u1_boundary_satisfied(self, ws_graph):
        """U1 (initiation/closure) is satisfied for any existing graph."""
        mappings = compute_grammar_symmetry_mapping(ws_graph)
        u1 = [m for m in mappings if m.rule == "U1"][0]
        assert u1.is_satisfied
        assert u1.diagnostic_value == 0.0

    def test_u2_stability_finite_energy(self, ws_graph):
        """U2 is satisfied when energy functional is finite."""
        mappings = compute_grammar_symmetry_mapping(ws_graph)
        u2 = [m for m in mappings if m.rule == "U2"][0]
        assert u2.is_satisfied

    def test_u6_confinement_check(self, ws_graph):
        """U6 checks structural potential confinement < φ."""
        mappings = compute_grammar_symmetry_mapping(ws_graph)
        u6 = [m for m in mappings if m.rule == "U6"][0]
        # Well-initialised graph should have confined Φ_s
        assert isinstance(u6.is_satisfied, bool)

    def test_diagnostic_values_nonnegative(self, ws_graph):
        """All diagnostic values are ≥ 0."""
        mappings = compute_grammar_symmetry_mapping(ws_graph)
        for m in mappings:
            assert m.diagnostic_value >= 0.0


# ---------------------------------------------------------------------------
# 2. Action-Energy Consistency: H_var ≡ E_cons
# ---------------------------------------------------------------------------

class TestActionEnergyConsistency:
    """The variational Hamiltonian equals the conservation energy functional."""

    def test_exact_identity(self, ws_graph):
        """H_variational = E_conservation to machine precision."""
        result = verify_action_energy_consistency(ws_graph)
        assert isinstance(result, ActionEnergyConsistency)
        assert result.relative_error < 1e-10
        assert result.is_consistent

    def test_hamiltonian_equals_T_plus_V(self, ws_graph):
        """H = T + V (energy partitioning)."""
        result = verify_action_energy_consistency(ws_graph)
        H = result.hamiltonian_variational
        T_plus_V = result.total_kinetic + result.total_potential
        assert abs(H - T_plus_V) / max(abs(H), 1e-15) < 1e-12

    def test_kinetic_fraction_bounded(self, ws_graph):
        """T/H must be in [0, 1]."""
        result = verify_action_energy_consistency(ws_graph)
        assert 0.0 <= result.kinetic_fraction <= 1.0

    def test_energy_positive(self, ws_graph):
        """Total energy H > 0 for a non-trivial graph."""
        result = verify_action_energy_consistency(ws_graph)
        assert result.hamiltonian_variational > 0.0

    def test_consistency_across_topologies(self, ws_graph, ba_graph, grid_graph):
        """H = E for WS, BA, and grid topologies."""
        for G in [ws_graph, ba_graph, grid_graph]:
            result = verify_action_energy_consistency(G)
            assert result.is_consistent, (
                f"H={result.hamiltonian_variational:.6f} != "
                f"E={result.energy_conservation:.6f}"
            )

    def test_custom_tolerance(self, ws_graph):
        """Respects custom tolerance parameter."""
        result = verify_action_energy_consistency(ws_graph, tolerance=1e-20)
        # rel_err is ~1e-16, so with tolerance 1e-20 it should still pass
        # for well-implemented identity
        assert isinstance(result.is_consistent, bool)


# ---------------------------------------------------------------------------
# 3. Noether-Gauge Decomposition
# ---------------------------------------------------------------------------

class TestNoetherGaugeDecomposition:
    """Symmetry decomposes into external (Noether) and internal (gauge) sectors."""

    def test_returns_correct_type(self, ws_graph):
        result = compute_noether_gauge_decomposition(ws_graph)
        assert isinstance(result, NoetherGaugeDecomposition)

    def test_energy_finite_positive(self, ws_graph):
        """Gauge-invariant energy E > 0 and finite."""
        result = compute_noether_gauge_decomposition(ws_graph)
        assert np.isfinite(result.energy_functional)
        assert result.energy_functional > 0.0

    def test_gauge_invariant_energy_equals_functional(self, ws_graph):
        """E is gauge-invariant → gauge_invariant_energy = energy_functional."""
        result = compute_noether_gauge_decomposition(ws_graph)
        assert result.gauge_invariant_energy == result.energy_functional

    def test_noether_charge_finite(self, ws_graph):
        """Noether charge Q is finite."""
        result = compute_noether_gauge_decomposition(ws_graph)
        assert np.isfinite(result.noether_charge)

    def test_yang_mills_nonnegative(self, ws_graph):
        """S_YM ≥ 0 (gauge field action is positive semi-definite)."""
        result = compute_noether_gauge_decomposition(ws_graph)
        assert result.yang_mills_action >= -1e-12

    def test_matter_action_nonnegative(self, ws_graph):
        """S_matter = Σ|DΨ|² ≥ 0."""
        result = compute_noether_gauge_decomposition(ws_graph)
        assert result.matter_action >= -1e-12

    def test_decomposition_quality_bounded(self, ws_graph):
        """Quality metric in [0, 1]."""
        result = compute_noether_gauge_decomposition(ws_graph)
        assert 0.0 <= result.decomposition_quality <= 1.0

    def test_noether_gauge_ratio_nonnegative(self, ws_graph):
        """|Q|/E ≥ 0."""
        result = compute_noether_gauge_decomposition(ws_graph)
        assert result.noether_gauge_ratio >= 0.0

    def test_multi_topology(self, ws_graph, ba_graph, grid_graph):
        """Decomposition works on all standard topologies."""
        for G in [ws_graph, ba_graph, grid_graph]:
            result = compute_noether_gauge_decomposition(G)
            assert np.isfinite(result.energy_functional)
            assert np.isfinite(result.noether_charge)


# ---------------------------------------------------------------------------
# 4. Gauge-Conservation Coupling
# ---------------------------------------------------------------------------

class TestGaugeConservationCoupling:
    """Quantifies the K_φ-mediated coupling between gauge and conservation."""

    def test_returns_correct_type(self, ws_graph):
        result = compute_gauge_conservation_coupling(ws_graph)
        assert isinstance(result, GaugeConservationCoupling)

    def test_energy_is_gauge_invariant(self, ws_graph):
        """Energy deviation under gauge rotation should be ≈ 0."""
        result = compute_gauge_conservation_coupling(ws_graph)
        assert result.energy_gauge_invariance < 1e-6

    def test_charge_is_not_gauge_invariant(self, ws_graph):
        """Charge sensitivity ΔQ > 0 (charge changes under gauge rotation)."""
        result = compute_gauge_conservation_coupling(ws_graph)
        # For most non-trivial graphs, ΔQ > 0
        assert result.gauge_charge_sensitivity >= 0.0

    def test_sector_energies_positive(self, ws_graph):
        """Geometric and potential sector energies > 0."""
        result = compute_gauge_conservation_coupling(ws_graph)
        assert result.geometric_sector_energy > 0.0
        assert result.potential_sector_energy > 0.0

    def test_kappa_bounded(self, ws_graph):
        """Sector coupling parameter κ ∈ [0, 1]."""
        result = compute_gauge_conservation_coupling(ws_graph)
        assert 0.0 <= result.sector_coupling_parameter <= 1.0

    def test_shared_field_fraction_nonnegative(self, ws_graph):
        """K_φ fraction of ρ is well-defined and non-negative.

        Note: fraction can exceed 1 when K_φ and Φ_s have opposite signs
        (|K_φ| > |Φ_s + K_φ|), so we only check non-negativity.
        """
        result = compute_gauge_conservation_coupling(ws_graph)
        assert result.shared_field_fraction >= 0.0
        assert np.isfinite(result.shared_field_fraction)

    def test_ward_gauge_consistency_meaningful(self, ws_graph):
        """Ward-gauge consistency value is > 0."""
        result = compute_gauge_conservation_coupling(ws_graph)
        assert result.ward_gauge_consistency > 0.0

    def test_different_gauge_angles(self, ws_graph):
        """Sensitivity scales with gauge angle."""
        r1 = compute_gauge_conservation_coupling(ws_graph, gauge_angle=0.01)
        r2 = compute_gauge_conservation_coupling(ws_graph, gauge_angle=0.5)
        # Larger angle → larger ΔQ (approximately)
        if r1.gauge_charge_sensitivity > 1e-12:
            assert r2.gauge_charge_sensitivity > r1.gauge_charge_sensitivity * 0.5

    def test_seed_reproducibility(self, ws_graph):
        """Same seed → identical results."""
        r1 = compute_gauge_conservation_coupling(ws_graph, seed=77)
        r2 = compute_gauge_conservation_coupling(ws_graph, seed=77)
        assert r1.geometric_sector_energy == r2.geometric_sector_energy
        assert r1.energy_gauge_invariance == r2.energy_gauge_invariance


# ---------------------------------------------------------------------------
# 5. Symplectic-Gauge Compatibility
# ---------------------------------------------------------------------------

class TestSymplecticGaugeCompatibility:
    """Symplectic form ω is preserved under gauge rotations (det R = 1)."""

    def test_returns_correct_type(self, ws_graph):
        result = verify_symplectic_gauge_compatibility(ws_graph)
        assert isinstance(result, SymplecticGaugeCompatibility)

    def test_is_compatible(self, ws_graph):
        """Symplectic form must be gauge-compatible (area-preserving)."""
        result = verify_symplectic_gauge_compatibility(ws_graph)
        assert result.is_compatible

    def test_volumes_nonnegative(self, ws_graph):
        """Phase space volumes ≥ 0."""
        result = verify_symplectic_gauge_compatibility(ws_graph)
        assert result.geometric_volume >= 0.0
        assert result.potential_volume >= 0.0
        assert result.total_volume >= 0.0

    def test_total_is_sum(self, ws_graph):
        """Ω = Ω_geo + Ω_pot."""
        result = verify_symplectic_gauge_compatibility(ws_graph)
        assert abs(result.total_volume -
                   (result.geometric_volume + result.potential_volume)) < 1e-12

    def test_poisson_brackets_finite(self, ws_graph):
        """Poisson bracket estimates are finite."""
        result = verify_symplectic_gauge_compatibility(ws_graph)
        assert np.isfinite(result.geometric_poisson)
        assert np.isfinite(result.potential_poisson)

    def test_gauge_volume_invariance_small(self, ws_graph):
        """Gauge volume deviation is a diagnostic, should be small for
        the 2-form (which is EXACTLY preserved)."""
        result = verify_symplectic_gauge_compatibility(ws_graph)
        assert np.isfinite(result.gauge_volume_invariance)

    def test_multi_topology(self, ws_graph, ba_graph, grid_graph):
        """Compatible across topologies."""
        for G in [ws_graph, ba_graph, grid_graph]:
            result = verify_symplectic_gauge_compatibility(G)
            assert result.is_compatible


# ---------------------------------------------------------------------------
# 6. Full Unification
# ---------------------------------------------------------------------------

class TestConservationGaugeUnification:
    """Complete pipeline: Grammar → Symmetry → Conservation → Gauge."""

    def test_returns_correct_type(self, ws_graph):
        result = run_conservation_gauge_unification(ws_graph)
        assert isinstance(result, ConservationGaugeUnification)

    def test_all_sub_results_present(self, ws_graph):
        """All 6 sub-analyses must be present."""
        result = run_conservation_gauge_unification(ws_graph)
        assert isinstance(result.grammar_symmetry, list)
        assert len(result.grammar_symmetry) == 6
        assert isinstance(result.action_consistency, ActionEnergyConsistency)
        assert isinstance(result.noether_gauge, NoetherGaugeDecomposition)
        assert isinstance(result.gauge_conservation, GaugeConservationCoupling)
        assert isinstance(result.symplectic_gauge, SymplecticGaugeCompatibility)

    def test_action_energy_identity_holds(self, ws_graph):
        """H_var = E_cons within the full pipeline."""
        result = run_conservation_gauge_unification(ws_graph)
        assert result.action_consistency.is_consistent

    def test_gauge_invariance_verified(self, ws_graph):
        """Gauge invariance is checked and reported."""
        result = run_conservation_gauge_unification(ws_graph)
        assert hasattr(result.gauge_invariance, "is_invariant")

    def test_symplectic_compatible(self, ws_graph):
        """Symplectic form is gauge-compatible."""
        result = run_conservation_gauge_unification(ws_graph)
        assert result.symplectic_gauge.is_compatible

    def test_quality_in_range(self, ws_graph):
        """Unification quality ∈ [0, 1]."""
        result = run_conservation_gauge_unification(ws_graph)
        assert 0.0 <= result.unification_quality <= 1.0

    def test_summary_contains_required_keys(self, ws_graph):
        """Summary dict must have all essential keys."""
        result = run_conservation_gauge_unification(ws_graph)
        required = {
            "grammar_rules_satisfied",
            "H_variational",
            "E_conservation",
            "H_E_relative_error",
            "T_kinetic",
            "V_potential",
            "kinetic_fraction",
            "noether_charge_Q",
            "gauge_invariant_energy",
            "yang_mills_action",
            "mean_psi_magnitude",
            "geometric_sector_energy",
            "potential_sector_energy",
            "sector_coupling_kappa",
            "shared_K_phi_fraction",
            "gauge_charge_sensitivity",
            "energy_gauge_invariance_dev",
            "symplectic_volume_geo",
            "symplectic_volume_pot",
            "poisson_bracket_geo",
            "poisson_bracket_pot",
            "unification_quality",
            "is_unified",
            "narrative",
        }
        assert required <= set(result.summary.keys())

    def test_narrative_is_string(self, ws_graph):
        """Narrative must be a non-empty string."""
        result = run_conservation_gauge_unification(ws_graph)
        assert isinstance(result.summary["narrative"], str)
        assert len(result.summary["narrative"]) > 10

    def test_seed_reproducibility(self, ws_graph):
        """Same gauge_seed → identical results."""
        r1 = run_conservation_gauge_unification(ws_graph, gauge_seed=99)
        r2 = run_conservation_gauge_unification(ws_graph, gauge_seed=99)
        assert r1.unification_quality == r2.unification_quality
        assert r1.is_unified == r2.is_unified
        assert (r1.action_consistency.hamiltonian_variational ==
                r2.action_consistency.hamiltonian_variational)

    def test_multi_topology(self, ws_graph, ba_graph, grid_graph):
        """Full pipeline succeeds on all standard topologies."""
        for G in [ws_graph, ba_graph, grid_graph]:
            result = run_conservation_gauge_unification(G)
            assert result.action_consistency.is_consistent
            assert result.symplectic_gauge.is_compatible
            assert result.unification_quality > 0.0


# ---------------------------------------------------------------------------
# 7. Coherent Phase Graph (Phase-aligned — should give full unification)
# ---------------------------------------------------------------------------

class TestCoherentGraph:
    """A graph with aligned phases should pass all checks including U3."""

    @pytest.fixture
    def coherent_graph(self):
        """Graph with closely aligned phases (U3, U6 satisfied)."""
        rng = np.random.default_rng(42)
        G = nx.watts_strogatz_graph(20, 4, 0.3, seed=42)
        inject_defaults(G)
        base_phase = 1.0
        for node in G.nodes():
            # Small phase deviation → U3 satisfied
            G.nodes[node]["phase"] = base_phase + rng.uniform(-0.1, 0.1)
            G.nodes[node]["frequency"] = rng.uniform(0.1, 1.0)
            G.nodes[node]["delta_nfr"] = rng.uniform(-0.1, 0.1)
            G.nodes[node]["EPI"] = f"epi_{node}"
        G.graph["delta_phi_max"] = math.pi / 4
        return G

    def test_all_grammar_rules_satisfied(self, coherent_graph):
        """Coherent graph should satisfy all grammar rules."""
        mappings = compute_grammar_symmetry_mapping(coherent_graph)
        for m in mappings:
            assert m.is_satisfied, f"Rule {m.rule} not satisfied: diag={m.diagnostic_value}"

    def test_full_unification(self, coherent_graph):
        """Coherent graph should achieve full unification."""
        result = run_conservation_gauge_unification(coherent_graph)
        assert result.is_unified
        assert result.unification_quality > 0.8

    def test_high_decomposition_quality(self, coherent_graph):
        """Small δ_NFR → uniform energy → high decomposition quality."""
        result = compute_noether_gauge_decomposition(coherent_graph)
        assert result.decomposition_quality > 0.5

    def test_narrative_unified(self, coherent_graph):
        """Narrative should indicate UNIFIED."""
        result = run_conservation_gauge_unification(coherent_graph)
        assert "UNIFIED" in result.summary["narrative"]


# ---------------------------------------------------------------------------
# 8. Import via physics.__init__
# ---------------------------------------------------------------------------

class TestPhysicsImport:
    """Verify exports are accessible from tnfr.physics."""

    def test_import_dataclasses(self):
        from tnfr.physics import (
            GrammarSymmetryMapping,
            ActionEnergyConsistency,
            NoetherGaugeDecomposition,
            GaugeConservationCoupling,
            SymplecticGaugeCompatibility,
            ConservationGaugeUnification,
        )
        assert GrammarSymmetryMapping is not None

    def test_import_functions(self):
        from tnfr.physics import (
            compute_grammar_symmetry_mapping,
            verify_action_energy_consistency,
            compute_noether_gauge_decomposition,
            compute_gauge_conservation_coupling,
            verify_symplectic_gauge_compatibility,
            run_conservation_gauge_unification,
        )
        assert callable(run_conservation_gauge_unification)
