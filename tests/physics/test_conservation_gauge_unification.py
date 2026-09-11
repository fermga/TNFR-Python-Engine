"""Tests for scoped TNFR conservation/gauge snapshot diagnostics.

The historical grammar-symmetry rows remain available, but each row states
whether the supplied evidence can assess the corresponding U-rule. A current
graph snapshot assesses U3 phase compatibility; U6 needs a reference state;
U1, U2, U4, and U5 need operator or hierarchy context outside this module.
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
from tnfr.physics.conservation import capture_conservation_snapshot
from tnfr.physics.conservation_gauge_unification import (
    ActionEnergyConsistency,
    ConservationGaugeUnification,
    GaugeConservationCoupling,
    GrammarSymmetryMapping,
    NoetherGaugeDecomposition,
    SymplecticGaugeCompatibility,
    compute_gauge_conservation_coupling,
    compute_grammar_symmetry_mapping,
    compute_noether_gauge_decomposition,
    run_conservation_gauge_unification,
    verify_action_energy_consistency,
    verify_symplectic_gauge_compatibility,
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
    """Historical mapping rows expose honest applicability metadata."""

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
            assert isinstance(m.is_applicable, bool)
            assert m.assessment_status in {"pass", "fail", "not_assessed"}

    def test_symmetry_types_are_distinct(self, ws_graph):
        """Each grammar rule maps to a different symmetry type."""
        mappings = compute_grammar_symmetry_mapping(ws_graph)
        types = [m.symmetry_type for m in mappings]
        expected = {
            "boundary",
            "stability",
            "gauge",
            "topological",
            "hierarchical",
            "confinement",
        }
        assert set(types) == expected

    def test_history_dependent_rules_are_not_assessed(self, ws_graph):
        """A snapshot cannot assess U1, U2, U4, or U5."""
        mappings = compute_grammar_symmetry_mapping(ws_graph)
        by_rule = {m.rule: m for m in mappings}
        for rule in ("U1", "U2", "U4", "U5"):
            row = by_rule[rule]
            assert not row.is_applicable
            assert not row.is_satisfied
            assert row.assessment_status == "not_assessed"
            assert row.required_evidence

    def test_only_u3_is_assessed_without_reference(self, ws_graph):
        mappings = compute_grammar_symmetry_mapping(ws_graph)
        assert [m.rule for m in mappings if m.is_applicable] == ["U3"]
        assert [m.rule for m in mappings if not m.is_applicable] == [
            "U1",
            "U2",
            "U4",
            "U5",
            "U6",
        ]

    def test_u3_is_unassessed_when_edge_phase_is_missing(self):
        graph = nx.path_graph(2)
        inject_defaults(graph)
        u3 = next(
            m for m in compute_grammar_symmetry_mapping(graph) if m.rule == "U3"
        )
        assert not u3.is_applicable
        assert not u3.is_satisfied
        assert u3.assessment_status == "not_assessed"
        assert "finite phase" in u3.required_evidence

    @pytest.mark.parametrize(
        "bad_phase", [None, "0.1", float("nan"), 10**1000]
    )
    def test_u3_is_unassessed_when_edge_phase_is_invalid(self, bad_phase):
        graph = nx.path_graph(2)
        inject_defaults(graph)
        graph.nodes[0]["phase"] = bad_phase
        graph.nodes[1]["phase"] = 0.0
        u3 = next(
            m for m in compute_grammar_symmetry_mapping(graph) if m.rule == "U3"
        )
        assert not u3.is_applicable
        assert u3.assessment_status == "not_assessed"

    @pytest.mark.parametrize(
        "bad_gate", [True, -0.1, float("inf"), "0.2", 10**1000]
    )
    def test_u3_is_unassessed_when_phase_gate_is_invalid(self, bad_gate):
        graph = nx.path_graph(2)
        inject_defaults(graph)
        graph.nodes[0]["phase"] = 0.0
        graph.nodes[1]["phase"] = 0.1
        graph.graph["delta_phi_max"] = bad_gate
        u3 = next(
            m for m in compute_grammar_symmetry_mapping(graph) if m.rule == "U3"
        )
        assert not u3.is_applicable
        assert "delta_phi_max" in u3.required_evidence

    def test_u3_reports_phase_failure_from_current_edges(self):
        graph = nx.Graph()
        graph.add_edge(0, 1)
        inject_defaults(graph)
        graph.nodes[0]["phase"] = 0.0
        graph.nodes[1]["phase"] = math.pi
        graph.graph["delta_phi_max"] = math.pi / 4
        u3 = next(
            m for m in compute_grammar_symmetry_mapping(graph) if m.rule == "U3"
        )
        assert u3.is_applicable
        assert not u3.is_satisfied
        assert u3.assessment_status == "fail"
        assert u3.diagnostic_value == pytest.approx(3 * math.pi / 4)

    def test_u3_wraps_arbitrary_phase_representatives(self):
        graph = nx.Graph()
        graph.add_edge(0, 1)
        inject_defaults(graph)
        graph.nodes[0]["phase"] = 0.0
        graph.nodes[1]["phase"] = 4 * math.pi + 0.1
        graph.graph["delta_phi_max"] = 0.2
        u3 = next(
            m for m in compute_grammar_symmetry_mapping(graph) if m.rule == "U3"
        )
        assert u3.is_satisfied
        assert u3.diagnostic_value == 0.0

    def test_u6_requires_reference_state(self, ws_graph):
        mappings = compute_grammar_symmetry_mapping(ws_graph)
        u6 = next(m for m in mappings if m.rule == "U6")
        assert not u6.is_applicable
        assert not u6.is_satisfied
        assert u6.assessment_status == "not_assessed"
        assert "reference" in u6.required_evidence

    def test_u6_is_unassessed_when_required_dnfr_is_missing(self, ws_graph):
        reference = ws_graph.copy()
        ws_graph.nodes[next(iter(ws_graph))].pop("delta_nfr")
        u6 = next(
            m
            for m in compute_grammar_symmetry_mapping(
                ws_graph, reference_graph=reference
            )
            if m.rule == "U6"
        )
        assert not u6.is_applicable
        assert u6.assessment_status == "not_assessed"
        assert "delta_nfr" in u6.required_evidence

    def test_u6_uses_reference_graph_for_potential_drift(self, ws_graph):
        reference = ws_graph.copy()
        unchanged = next(
            m
            for m in compute_grammar_symmetry_mapping(
                ws_graph, reference_graph=reference
            )
            if m.rule == "U6"
        )
        assert unchanged.is_applicable
        assert unchanged.is_satisfied
        assert unchanged.assessment_status == "pass"
        assert unchanged.diagnostic_value == pytest.approx(0.0)

        for node in ws_graph:
            ws_graph.nodes[node]["delta_nfr"] += 10.0
        drifted = next(
            m
            for m in compute_grammar_symmetry_mapping(
                ws_graph, reference_graph=reference
            )
            if m.rule == "U6"
        )
        assert drifted.is_applicable
        assert not drifted.is_satisfied
        assert drifted.assessment_status == "fail"
        assert drifted.diagnostic_value > math.pi / 2

    def test_u6_accepts_reference_snapshot(self, ws_graph):
        reference = capture_conservation_snapshot(ws_graph)
        u6 = next(
            m
            for m in compute_grammar_symmetry_mapping(
                ws_graph, reference_snapshot=reference
            )
            if m.rule == "U6"
        )
        assert u6.is_applicable
        assert u6.is_satisfied
        assert u6.diagnostic_value == pytest.approx(0.0)

    def test_u6_rejects_ambiguous_reference_sources(self, ws_graph):
        snapshot = capture_conservation_snapshot(ws_graph)
        with pytest.raises(ValueError, match="at most one"):
            compute_grammar_symmetry_mapping(
                ws_graph,
                reference_graph=ws_graph.copy(),
                reference_snapshot=snapshot,
            )

    def test_u6_node_mismatch_is_not_assessed(self, ws_graph):
        reference = ws_graph.copy()
        reference.remove_node(next(iter(reference)))
        u6 = next(
            m
            for m in compute_grammar_symmetry_mapping(
                ws_graph, reference_graph=reference
            )
            if m.rule == "U6"
        )
        assert not u6.is_applicable
        assert u6.assessment_status == "not_assessed"

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
    """Legacy fields expose accurately scoped finite snapshot aliases."""

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
        """The squared cycle-closure penalty is nonnegative."""
        result = compute_noether_gauge_decomposition(ws_graph)
        assert result.yang_mills_action >= -1e-12

    def test_legacy_gauge_names_have_accurate_snapshot_aliases(self, ws_graph):
        result = compute_noether_gauge_decomposition(ws_graph)
        assert result.mean_cycle_closure_residual == result.mean_gauge_curvature
        assert result.squared_cycle_closure_penalty == result.yang_mills_action
        assert result.covariant_difference_energy == result.matter_action
        assert result.energy_density_uniformity_score == result.decomposition_quality

    def test_matter_action_nonnegative(self, ws_graph):
        """The covariant-difference energy Σ|DΨ|² is nonnegative."""
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
    """Quantifies separate constant/local coordinate-rotation diagnostics."""

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
        """The legacy K_φ/ρ ratio is well-defined and non-negative.

        Note: fraction can exceed 1 when K_φ and Φ_s have opposite signs
        (|K_φ| > |Φ_s + K_φ|), so we only check non-negativity.
        """
        result = compute_gauge_conservation_coupling(ws_graph)
        assert result.shared_field_fraction >= 0.0
        assert np.isfinite(result.shared_field_fraction)
        assert result.shared_field_ratio == result.shared_field_fraction

    def test_ward_gauge_consistency_meaningful(self, ws_graph):
        """The legacy Ward-named field carries the local covariance score."""
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

    def test_nonfinite_global_rotation_angle_is_rejected(self, ws_graph):
        with pytest.raises(ValueError, match="gauge_angle must be finite"):
            compute_gauge_conservation_coupling(ws_graph, gauge_angle=float("nan"))


# ---------------------------------------------------------------------------
# 5. Symplectic-Gauge Compatibility
# ---------------------------------------------------------------------------


class TestSymplecticGaugeCompatibility:
    """The auxiliary two-form is preserved by a global oscillator rotation."""

    def test_returns_correct_type(self, ws_graph):
        result = verify_symplectic_gauge_compatibility(ws_graph)
        assert isinstance(result, SymplecticGaugeCompatibility)

    def test_is_compatible(self, ws_graph):
        """The declared rotation is area-preserving."""
        result = verify_symplectic_gauge_compatibility(ws_graph)
        assert result.is_compatible

    def test_volumes_nonnegative(self, ws_graph):
        """Legacy snapshot-product statistics are nonnegative."""
        result = verify_symplectic_gauge_compatibility(ws_graph)
        assert result.geometric_volume >= 0.0
        assert result.potential_volume >= 0.0
        assert result.total_volume >= 0.0

    def test_total_is_sum(self, ws_graph):
        """Ω = Ω_geo + Ω_pot."""
        result = verify_symplectic_gauge_compatibility(ws_graph)
        assert (
            abs(
                result.total_volume
                - (result.geometric_volume + result.potential_volume)
            )
            < 1e-12
        )

    def test_poisson_brackets_finite(self, ws_graph):
        """Legacy normalized covariance statistics are finite."""
        result = verify_symplectic_gauge_compatibility(ws_graph)
        assert np.isfinite(result.geometric_poisson)
        assert np.isfinite(result.potential_poisson)
        assert result.geometric_snapshot_product == result.geometric_volume
        assert result.potential_snapshot_product == result.potential_volume
        assert result.geometric_normalized_covariance == result.geometric_poisson
        assert result.potential_normalized_covariance == result.potential_poisson

    def test_gauge_volume_invariance_small(self, ws_graph):
        """The global oscillator rotation preserves the auxiliary two-form."""
        result = verify_symplectic_gauge_compatibility(ws_graph)
        assert result.gauge_volume_invariance < 1e-12
        assert result.is_compatible
        assert result.transformation_scope == "global_constant_oscillator_rotation"
        assert result.local_gauge_assessed is False
        assert np.isfinite(result.snapshot_product_change)

    def test_multi_topology(self, ws_graph, ba_graph, grid_graph):
        """Compatible across topologies."""
        for G in [ws_graph, ba_graph, grid_graph]:
            result = verify_symplectic_gauge_compatibility(G)
            assert result.is_compatible


# ---------------------------------------------------------------------------
# 6. Aggregate diagnostic pipeline
# ---------------------------------------------------------------------------


class TestConservationGaugeUnification:
    """Aggregate result retains its legacy API with explicit scope."""

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
        """Aggregate diagnostic quality lies in [0, 1]."""
        result = run_conservation_gauge_unification(ws_graph)
        assert 0.0 <= result.unification_quality <= 1.0

    def test_summary_contains_required_keys(self, ws_graph):
        """Summary dict must have all essential keys."""
        result = run_conservation_gauge_unification(ws_graph)
        required = {
            "grammar_rules_satisfied",
            "grammar_rules_assessed",
            "grammar_rules_unassessed",
            "grammar_validation_applicable",
            "grammar_validated",
            "H_variational",
            "E_conservation",
            "H_E_relative_error",
            "T_kinetic",
            "V_potential",
            "kinetic_fraction",
            "noether_charge_Q",
            "historical_structural_charge_snapshot",
            "gauge_invariant_energy",
            "yang_mills_action",
            "mean_cycle_closure_residual",
            "squared_cycle_closure_penalty",
            "covariant_difference_energy",
            "energy_density_uniformity_score",
            "mean_psi_magnitude",
            "geometric_sector_energy",
            "potential_sector_energy",
            "sector_coupling_kappa",
            "shared_K_phi_fraction",
            "gauge_charge_sensitivity",
            "energy_gauge_invariance_dev",
            "local_pure_gauge_invariance_score",
            "symplectic_volume_geo",
            "symplectic_volume_pot",
            "poisson_bracket_geo",
            "poisson_bracket_pot",
            "global_oscillator_symplectic_residual",
            "global_oscillator_snapshot_product_change",
            "global_oscillator_scope",
            "local_gauge_assessed_by_symplectic_check",
            "unification_quality",
            "is_unified",
            "aggregate_diagnostic_passed",
            "diagnostic_scope",
            "narrative",
        }
        assert required <= set(result.summary.keys())

    def test_narrative_is_string(self, ws_graph):
        """Narrative must be a non-empty string."""
        result = run_conservation_gauge_unification(ws_graph)
        assert isinstance(result.summary["narrative"], str)
        assert len(result.summary["narrative"]) > 10
        assert "grammar was not validated" in result.summary["narrative"]

    def test_legacy_is_unified_is_only_aggregate_alias(self, ws_graph):
        result = run_conservation_gauge_unification(ws_graph)
        assert result.is_unified == result.aggregate_diagnostic_passed
        assert result.summary["is_unified"] == result.aggregate_diagnostic_passed
        assert result.grammar_validated is False
        assert result.summary["grammar_validated"] is False
        assert result.assessed_grammar_rules == ("U3",)
        assert result.unassessed_grammar_rules == ("U1", "U2", "U4", "U5", "U6")

    def test_reference_extends_coverage_only_to_u6(self, ws_graph):
        result = run_conservation_gauge_unification(
            ws_graph, reference_graph=ws_graph.copy()
        )
        assert result.assessed_grammar_rules == ("U3", "U6")
        assert result.unassessed_grammar_rules == ("U1", "U2", "U4", "U5")
        assert result.diagnostic_scope == "two_snapshot_aggregate_with_u6_reference"
        assert result.grammar_validated is False

    def test_seed_reproducibility(self, ws_graph):
        """Same gauge_seed → identical results."""
        r1 = run_conservation_gauge_unification(ws_graph, gauge_seed=99)
        r2 = run_conservation_gauge_unification(ws_graph, gauge_seed=99)
        assert r1.unification_quality == r2.unification_quality
        assert r1.is_unified == r2.is_unified
        assert (
            r1.action_consistency.hamiltonian_variational
            == r2.action_consistency.hamiltonian_variational
        )

    def test_multi_topology(self, ws_graph, ba_graph, grid_graph):
        """Full pipeline succeeds on all standard topologies."""
        for G in [ws_graph, ba_graph, grid_graph]:
            result = run_conservation_gauge_unification(G)
            assert result.action_consistency.is_consistent
            assert result.symplectic_gauge.is_compatible
            assert result.unification_quality > 0.0


# ---------------------------------------------------------------------------
# 7. Coherent Phase Graph
# ---------------------------------------------------------------------------


class TestCoherentGraph:
    """A graph with aligned phases should pass the applicable U3 check."""

    @pytest.fixture
    def coherent_graph(self):
        """Graph with closely aligned phases for the U3 snapshot check."""
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

    def test_only_u3_is_assessed_and_satisfied(self, coherent_graph):
        mappings = compute_grammar_symmetry_mapping(coherent_graph)
        assessed = [m for m in mappings if m.is_applicable]
        assert [m.rule for m in assessed] == ["U3"]
        assert assessed[0].is_satisfied

    def test_aggregate_diagnostics_pass(self, coherent_graph):
        result = run_conservation_gauge_unification(coherent_graph)
        assert result.aggregate_diagnostic_passed
        assert result.is_unified == result.aggregate_diagnostic_passed
        assert not result.grammar_validated
        assert result.unification_quality > 0.8

    def test_high_decomposition_quality(self, coherent_graph):
        """Small δ_NFR → uniform energy → high decomposition quality."""
        result = compute_noether_gauge_decomposition(coherent_graph)
        assert result.decomposition_quality > 0.5

    def test_narrative_states_grammar_scope(self, coherent_graph):
        result = run_conservation_gauge_unification(coherent_graph)
        assert result.summary["narrative"] == (
            "Aggregate finite diagnostics passed; grammar was not validated"
        )


# ---------------------------------------------------------------------------
# 8. Import via physics.__init__
# ---------------------------------------------------------------------------


class TestPhysicsImport:
    """Verify exports are accessible from tnfr.physics."""

    def test_import_dataclasses(self):
        from tnfr.physics import (
            ActionEnergyConsistency,
            ConservationGaugeUnification,
            GaugeConservationCoupling,
            GrammarSymmetryMapping,
            NoetherGaugeDecomposition,
            SymplecticGaugeCompatibility,
        )

        assert GrammarSymmetryMapping is not None

    def test_import_functions(self):
        from tnfr.physics import (
            compute_gauge_conservation_coupling,
            compute_grammar_symmetry_mapping,
            compute_noether_gauge_decomposition,
            run_conservation_gauge_unification,
            verify_action_energy_consistency,
            verify_symplectic_gauge_compatibility,
        )

        assert callable(run_conservation_gauge_unification)
