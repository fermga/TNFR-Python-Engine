"""Tests for U2 policy multipliers and independent spectral diagnostics.

Validates:
1. Canonical U2 roles and centralized default-factor provenance
2. Spectral gap characterisation (SpectralGapAnalysis)
3. Legacy API compatibility without claiming energy bounds
4. Nominal sequence-multiplier composition
5. Separation of operator-position policy and continuous-time diffusion scales

The five-field energy is a candidate diagnostic.  U2 role labels and configured
glyph factors do not imply numerical bounds on its change.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from tnfr.constants import inject_defaults
from tnfr.config.defaults_core import CORE_DEFAULTS
from tnfr.physics.lyapunov import (
    OPERATOR_POLICY_MULTIPLIERS,
    OPERATOR_LYAPUNOV_BOUNDS,
    EnergyClass,
    U2PolicyRole,
    OperatorLyapunovBound,
    analyze_operator_policy_context,
    analyze_operator_convergence,
    analyze_spectral_gap,
    compare_operator_energy_to_policy,
    compute_operator_policy_delta,
    compute_operator_energy_bound,
    compute_sequence_policy_score,
    compute_sequence_energy_bound,
    evaluate_sequence_policy,
    get_bound,
    get_policy_multiplier,
    prove_sequence_lyapunov,
    verify_operator_lyapunov,
)
from tnfr.physics.structural_diffusion import structural_diffusion_operator

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
# 1. Operator policy-multiplier registry
# ===========================================================================


class TestOperatorLyapunovBoundsRegistry:
    """All 13 operators have one centralized policy entry."""

    EXPECTED_OPERATORS = [
        "Coherence",
        "Reception",
        "Coupling",
        "SelfOrganization",
        "Transition",
        "Dissonance",
        "Expansion",
        "Emission",
        "Resonance",
        "Silence",
        "Mutation",
        "Recursivity",
        "Contraction",
    ]

    EXPECTED_GLYPHS = [
        "IL",
        "EN",
        "UM",
        "THOL",
        "NAV",
        "OZ",
        "VAL",
        "AL",
        "RA",
        "SHA",
        "ZHIR",
        "REMESH",
        "NUL",
    ]

    def test_registry_has_13_operators(self):
        assert len(OPERATOR_LYAPUNOV_BOUNDS) == 13
        assert OPERATOR_POLICY_MULTIPLIERS is OPERATOR_LYAPUNOV_BOUNDS

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
        assert get_policy_multiplier(glyph) is bound

    def test_unknown_operator_raises(self):
        with pytest.raises(KeyError):
            get_bound("NonExistent")

    def test_all_have_scoped_policy_rationale(self):
        for name, bound in OPERATOR_LYAPUNOV_BOUNDS.items():
            assert len(bound.derivation) > 10, f"{name} missing derivation"
            assert bound.verification_scope == "u2_policy_heuristic"
            assert not bound.is_energy_bound

    def test_all_have_positive_glyph_factor(self):
        for name, bound in OPERATOR_LYAPUNOV_BOUNDS.items():
            assert bound.glyph_factor_value >= 0.0, f"{name} invalid factor"

    def test_factor_values_come_from_canonical_defaults(self):
        factors = CORE_DEFAULTS["GLYPH_FACTORS"]
        for bound in OPERATOR_POLICY_MULTIPLIERS.values():
            assert bound.glyph_factor_value == pytest.approx(
                factors[bound.glyph_factor_name]
            )


# ===========================================================================
# 2. Energy class taxonomy
# ===========================================================================


class TestEnergyClassTaxonomy:
    """The legacy EnergyClass field mirrors U2 roles, not energy signs."""

    STABILISERS = ["Coherence", "SelfOrganization"]
    DESTABILISERS = ["Dissonance", "Expansion", "Mutation"]
    NEUTRALS = [
        "Emission",
        "Reception",
        "Resonance",
        "Coupling",
        "Silence",
        "Contraction",
        "Transition",
        "Recursivity",
    ]

    def test_policy_named_enum_alias_preserves_legacy_identity(self):
        assert U2PolicyRole is EnergyClass

    @pytest.mark.parametrize("name", STABILISERS)
    def test_stabilisers(self, name):
        assert get_bound(name).energy_class == EnergyClass.STABILISER

    @pytest.mark.parametrize("name", DESTABILISERS)
    def test_destabilisers(self, name):
        assert get_bound(name).energy_class == EnergyClass.DESTABILISER

    @pytest.mark.parametrize("name", NEUTRALS)
    def test_neutrals(self, name):
        assert get_bound(name).energy_class == EnergyClass.NEUTRAL

    def test_classification_matches_canonical_grammar(self):
        """The legacy class field derives from the centralized U2 predicates."""
        from tnfr.config.physics_derivation import (
            increases_structural_pressure,
            provides_negative_feedback,
        )

        name_to_func = {
            "Emission": "emission",
            "Reception": "reception",
            "Coherence": "coherence",
            "Dissonance": "dissonance",
            "Coupling": "coupling",
            "Resonance": "resonance",
            "Silence": "silence",
            "Expansion": "expansion",
            "Contraction": "contraction",
            "SelfOrganization": "self_organization",
            "Mutation": "mutation",
            "Transition": "transition",
            "Recursivity": "recursivity",
        }
        for name, func in name_to_func.items():
            cls = get_bound(name).energy_class
            if provides_negative_feedback(func):
                assert cls == EnergyClass.STABILISER, name
            elif increases_structural_pressure(func):
                assert cls == EnergyClass.DESTABILISER, name
            else:
                assert cls == EnergyClass.NEUTRAL, name


# ===========================================================================
# 3. Per-operator policy adjustments
# ===========================================================================


class TestContractionRates:
    """Verify legacy rate fields encode the declared multiplier policy."""

    def test_coherence_policy_uses_canonical_retention(self):
        bound = get_bound("IL")
        assert bound.policy_multiplier == pytest.approx(
            CORE_DEFAULTS["GLYPH_FACTORS"]["IL_dnfr_factor"]
        )
        assert bound.policy_rate == pytest.approx(1.0 - bound.policy_multiplier)

    def test_dissonance_policy_uses_canonical_factor(self):
        bound = get_bound("OZ")
        assert bound.policy_multiplier == pytest.approx(
            CORE_DEFAULTS["GLYPH_FACTORS"]["OZ_dnfr_factor"]
        )

    def test_emission_rate_small(self):
        """AL has no U2 debt adjustment; this says nothing about energy."""
        bound = get_bound("AL")
        assert bound.contraction_rate == 0.0

    def test_resonance_rate_approx_01(self):
        """RA has no U2 debt adjustment."""
        bound = get_bound("RA")
        assert bound.contraction_rate == 0.0

    def test_expansion_policy_uses_capacity_scale(self):
        bound = get_bound("VAL")
        assert bound.policy_multiplier == pytest.approx(
            CORE_DEFAULTS["GLYPH_FACTORS"]["VAL_scale"]
        )

    def test_silence_rate_small(self):
        """SHA has no U2 debt adjustment."""
        bound = get_bound("SHA")
        assert bound.contraction_rate == 0.0

    def test_recursivity_rate_zero(self):
        """REMESH is U2-neutral; no zero-energy claim follows."""
        bound = get_bound("REMESH")
        assert bound.contraction_rate == 0.0

    def test_stabilisers_have_positive_rates(self):
        # Positive adjustments define sub-unit policy multipliers only.
        for name in ["Coherence", "SelfOrganization"]:
            assert get_bound(name).contraction_rate > 0.0
            assert get_bound(name).policy_multiplier < 1.0


# ===========================================================================
# 4. Policy-score computation and legacy compatibility
# ===========================================================================


class TestComputeOperatorEnergyBound:
    """The historical API computes a policy-score delta, not an energy bound."""

    def test_stabiliser_policy_delta_is_negative(self):
        e0 = 10.0
        delta = compute_operator_policy_delta("Coherence", e0, n_nodes=20)
        assert delta < 0.0

    def test_destabiliser_policy_delta_is_positive(self):
        e0 = 10.0
        delta = compute_operator_policy_delta("Dissonance", e0, n_nodes=20)
        assert delta > 0.0

    def test_legacy_wrapper_matches_policy_api(self):
        assert compute_operator_energy_bound("IL", 10.0, 20) == pytest.approx(
            compute_operator_policy_delta("IL", 10.0, 20)
        )

    def test_u2_neutral_delta_is_zero_independent_of_node_count(self):
        d5 = compute_operator_policy_delta("Silence", 10.0, n_nodes=5)
        d1 = compute_operator_policy_delta("Silence", 10.0, n_nodes=1)
        assert d5 == d1 == 0.0

    def test_recursivity_bound_is_zero(self):
        delta = compute_operator_energy_bound("Recursivity", 10.0, n_nodes=20)
        assert delta == 0.0

    def test_zero_score_gives_zero_policy_delta(self):
        assert compute_operator_energy_bound("Coherence", 0.0) == 0.0
        assert compute_operator_energy_bound("Dissonance", 0.0) == 0.0

    @pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan"), True])
    def test_invalid_policy_score_rejected(self, value):
        with pytest.raises(ValueError):
            compute_operator_policy_delta("IL", value)

    @pytest.mark.parametrize("n_nodes", [0, -1, 1.5, True])
    def test_invalid_node_count_rejected(self, n_nodes):
        with pytest.raises(ValueError):
            compute_operator_policy_delta("IL", 1.0, n_nodes=n_nodes)


# ===========================================================================
# 5. Measured-energy versus policy comparison
# ===========================================================================


class TestVerifyOperatorLyapunov:
    """The legacy boolean is explicitly a policy screen, never a certificate."""

    def test_coherence_can_pass_nominal_one_sided_screen(self):
        result = compare_operator_energy_to_policy("IL", 10.0, 4.0, n_nodes=20)
        assert result.policy_screen_passed
        assert result.delta_e < 0.0
        assert result.margin > 0.0
        assert not result.is_lyapunov_certificate

    def test_dissonance_can_pass_nominal_one_sided_screen(self):
        e0 = 1.0
        result = compare_operator_energy_to_policy("OZ", e0, 1.2, n_nodes=20)
        assert result.within_bound

    def test_dissonance_can_exceed_policy_screen(self):
        e0 = 1.0
        result = verify_operator_lyapunov("OZ", e0, 100.0, n_nodes=20)
        assert not result.within_bound

    def test_legacy_and_policy_names_are_consistent(self):
        result = verify_operator_lyapunov("IL", 10.0, 8.0, n_nodes=10)
        assert result.operator_name == "Coherence"
        assert result.glyph == "IL"
        assert result.energy_class == EnergyClass.STABILISER
        assert abs(result.delta_e - (-2.0)) < 1e-12
        assert result.policy_delta == result.theoretical_bound
        assert result.policy_residual == pytest.approx(
            result.delta_e - result.policy_delta
        )
        assert result.observed_energy_ratio == pytest.approx(0.8)
        assert result.multiplier_residual == pytest.approx(
            result.observed_energy_ratio - result.policy_multiplier
        )
        assert result.verification_scope == "measured_energy_vs_u2_policy_heuristic"

    def test_legacy_wrapper_matches_policy_comparison(self):
        assert verify_operator_lyapunov("IL", 3.0, 2.0) == (
            compare_operator_energy_to_policy("IL", 3.0, 2.0)
        )

    def test_zero_baseline_has_no_observed_ratio(self):
        result = compare_operator_energy_to_policy("AL", 0.0, 0.0)
        assert math.isnan(result.observed_energy_ratio)
        assert math.isnan(result.multiplier_residual)

    @pytest.mark.parametrize("tolerance", [-1.0, float("inf"), float("nan")])
    def test_invalid_comparison_tolerance_rejected(self, tolerance):
        with pytest.raises(ValueError):
            compare_operator_energy_to_policy("IL", 1.0, 1.0, tolerance=tolerance)


# ===========================================================================
# 6. Sequence policy score
# ===========================================================================


class TestSequenceEnergyBound:
    """Test multiplier composition on an abstract score."""

    def test_pure_stabiliser_sequence_reduces_policy_score(self):
        score = compute_sequence_policy_score(["IL", "IL", "IL"], 10.0, 20)
        assert score < 10.0

    def test_pure_destabiliser_sequence_raises_policy_score(self):
        score = compute_sequence_policy_score(["OZ", "OZ"], 1.0, 20)
        assert score > 1.0

    def test_legacy_wrapper_matches_policy_score(self):
        sequence = ["AL", "UM", "IL"]
        assert compute_sequence_energy_bound(sequence, 1.0, 20) == pytest.approx(
            compute_sequence_policy_score(sequence, 1.0, 20)
        )

    def test_policy_score_is_finite_for_finite_fragment(self):
        score = compute_sequence_policy_score(["OZ", "ZHIR", "IL"], 1.0, 20)
        assert math.isfinite(score)

    @pytest.mark.parametrize("operators", ["IL", ["IL", 1]])
    def test_invalid_operator_sequence_rejected(self, operators):
        with pytest.raises(TypeError):
            compute_sequence_policy_score(operators, 1.0)


# ===========================================================================
# 7. Sequence policy evaluation
# ===========================================================================


class TestSequenceLyapunovProof:
    """Test the legacy record without interpreting it as a proof."""

    def test_stabiliser_only_has_subunit_policy_product(self):
        result = evaluate_sequence_policy(["IL", "IL", "IL"])
        assert result.policy_product_at_most_one
        assert result.cumulative_product < 1.0
        assert result.net_contraction > 0.0
        assert not result.is_lyapunov_proof

    def test_reciprocal_oz_il_defaults_cancel_in_policy_model(self):
        result = evaluate_sequence_policy(["OZ", "IL"])
        assert result.cumulative_product == pytest.approx(1.0)
        assert result.policy_product_at_most_one

    def test_additional_il_reduces_policy_product(self):
        result = evaluate_sequence_policy(["OZ", "IL", "IL"])
        assert result.cumulative_product < 1.0

    def test_sequence_evaluator_does_not_claim_grammar_validation(self):
        seq = ["AL", "UM", "IL", "OZ", "ZHIR", "IL", "IL", "IL", "IL", "SHA"]
        result = evaluate_sequence_policy(seq)
        assert len(result.operators) == 10
        assert len(result.policy_multipliers) == 10
        assert result.verification_scope == "u2_policy_multiplier_product"

    def test_legacy_wrapper_matches_policy_api(self):
        assert prove_sequence_lyapunov(["IL"]) == evaluate_sequence_policy(["IL"])

    def test_plain_string_is_not_accepted_as_a_sequence(self):
        with pytest.raises(TypeError):
            evaluate_sequence_policy("IL")


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

    def test_relaxation_time_is_inverse_normalized_diffusion_gap(self, ws_graph):
        result = analyze_spectral_gap(ws_graph)
        expected_tau = 1.0 / result.diffusion_gap
        assert abs(result.relaxation_time - expected_tau) < 1e-10
        assert result.convergence_rate == result.diffusion_gap

    def test_legacy_mixing_scale_uses_normalized_gap(self, ws_graph):
        result = analyze_spectral_gap(ws_graph)
        n = ws_graph.number_of_nodes()
        expected = math.log(n) / result.diffusion_gap
        assert abs(result.mixing_time_bound - expected) < 1e-10

    def test_normalized_cheeger_lower_expression(self, ws_graph):
        result = analyze_spectral_gap(ws_graph)
        assert result.cheeger_lower == pytest.approx(0.5 * result.diffusion_gap)

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

    def test_disconnected_graph_has_zero_gaps(self):
        G = nx.disjoint_union(nx.path_graph(3), nx.path_graph(2))
        result = analyze_spectral_gap(G)
        assert not result.is_connected
        assert result.spectral_gap == 0.0
        assert result.diffusion_gap == 0.0
        assert math.isinf(result.relaxation_time)

    def test_connectivity_is_invariant_under_tiny_global_weight_scale(self):
        G = nx.path_graph(4)
        nx.set_edge_attributes(G, 1e-12, "weight")
        result = analyze_spectral_gap(G)
        assert result.is_connected
        assert result.spectral_gap > 0.0
        assert result.diffusion_gap > 0.0
        assert math.isfinite(result.relaxation_time)

    @pytest.mark.parametrize(
        "topology", ["watts_strogatz", "barabasi_albert", "complete"]
    )
    def test_positive_gap_across_topologies(self, topology):
        G = _make_tnfr_graph(15, topology, seed=7)
        result = analyze_spectral_gap(G)
        assert result.spectral_gap > 0.0
        assert result.is_connected


class TestDiffusionHTheoremRate:
    """The proven diffusion H-theorem relaxes at the canonical diffusion_gap rate.

    Connects the structural H-theorem (Dirichlet energy of the EPI diffusion
    channel, example 135) to the conservation/Lyapunov relaxation rate: the
    canonical relaxation eigenvalue is λ₂(L_sym) = diffusion_gap (the Fiedler
    eigenvalue of L_rw), NOT the combinatorial λ₂(L = D − A).  The Dirichlet
    energy decays at 2·νf·diffusion_gap.  See STRUCTURAL_CONSERVATION_THEOREM §8.6.
    """

    def test_diffusion_gap_is_the_h_theorem_relaxation_rate(self):
        G = nx.barabasi_albert_graph(30, 2, seed=3)  # irregular: the gaps differ
        inject_defaults(G)
        nodes, L_rw = structural_diffusion_operator(G)
        L_rw = np.asarray(L_rw, dtype=float)
        A = nx.to_numpy_array(G, nodelist=nodes)
        deg = A.sum(1)
        L_comb = np.diag(deg) - A
        L_sym = nx.normalized_laplacian_matrix(G, nodelist=nodes).toarray()
        vals, vecs = np.linalg.eigh(L_sym)
        order = np.argsort(vals)
        lam2 = float(vals[order[1]])
        d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        v2 = d_inv_sqrt * vecs[:, order[1]]  # L_rw eigenvector for λ₂

        sg = analyze_spectral_gap(G)
        # diffusion_gap == λ₂(L_sym), the canonical relaxation eigenvalue
        assert abs(sg.diffusion_gap - lam2) < 1e-9
        # it is the Fiedler eigenvalue of the canonical operator L_rw itself
        assert np.linalg.norm(
            L_rw @ v2 - sg.diffusion_gap * v2
        ) < 1e-9 * np.linalg.norm(v2)
        # and NOT the combinatorial gap on this irregular graph
        assert abs(sg.diffusion_gap - sg.spectral_gap) > 1e-3

        # the diffusion H-theorem's Dirichlet energy decays at 2·νf·diffusion_gap:
        # one explicit Euler step of ∂EPI/∂t = −νf·L_rw·EPI on the λ₂ mode
        vf, dt = 0.7, 0.02
        f0 = float(v2 @ L_comb @ v2)
        epi_dt = v2 - dt * vf * (L_rw @ v2)
        f_dt = float(epi_dt @ L_comb @ epi_dt)
        assert abs(f_dt / f0 - (1.0 - dt * vf * sg.diffusion_gap) ** 2) < 1e-9


# ===========================================================================
# 9. Independent policy and spectral context
# ===========================================================================


class TestOperatorConvergence:
    """The API no longer fabricates a rate from incompatible clocks."""

    def test_stabiliser_reports_policy_half_steps_and_diffusion_time(self, ws_graph):
        summary = analyze_operator_policy_context(ws_graph, "IL")
        assert 0.0 < summary.policy_multiplier < 1.0
        assert math.isfinite(summary.policy_half_steps)
        assert summary.diffusion_relaxation_time == summary.spectral.relaxation_time
        assert not summary.combination_defined

    def test_destabiliser_has_no_policy_half_steps(self, ws_graph):
        summary = analyze_operator_convergence(ws_graph, "OZ")
        assert summary.policy_multiplier > 1.0
        assert summary.policy_half_steps == float("inf")

    def test_effective_rate_is_explicitly_undefined(self, ws_graph):
        summary = analyze_operator_convergence(ws_graph, "IL")
        assert math.isnan(summary.effective_convergence_rate)
        assert summary.verification_scope == (
            "independent_policy_and_pure_epi_spectral_readouts"
        )

    def test_policy_half_steps_use_discrete_multiplier(self, ws_graph):
        summary = analyze_operator_convergence(ws_graph, "IL")
        expected = math.log(0.5) / math.log(summary.policy_multiplier)
        assert summary.policy_half_steps == pytest.approx(expected)
        assert summary.steps_to_half_energy == summary.policy_half_steps

    @pytest.mark.parametrize("glyph", ["IL", "THOL"])
    def test_all_stabilisers_have_subunit_policy_multipliers(self, ws_graph, glyph):
        summary = analyze_operator_convergence(ws_graph, glyph)
        assert summary.policy_multiplier < 1.0
        assert not summary.combination_defined

    def test_legacy_wrapper_matches_policy_context(self, ws_graph):
        legacy = analyze_operator_convergence(ws_graph, "IL")
        preferred = analyze_operator_policy_context(ws_graph, "IL")
        assert legacy.operator_bound == preferred.operator_bound
        assert legacy.policy_multiplier == preferred.policy_multiplier
        assert legacy.policy_half_steps == preferred.policy_half_steps
        assert np.array_equal(
            legacy.spectral.eigenvalues, preferred.spectral.eigenvalues
        )


# ===========================================================================
# 10. Edge cases and robustness
# ===========================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_initial_energy(self):
        """A zero policy score should not cause division errors."""
        for glyph in ["IL", "OZ", "AL", "SHA", "REMESH"]:
            d = compute_operator_energy_bound(glyph, 0.0, n_nodes=10)
            assert math.isfinite(d)

    def test_very_small_energy(self):
        """A very small policy score should produce a small model delta."""
        d = compute_operator_energy_bound("IL", 1e-15, n_nodes=1)
        assert abs(d) < 1e-14

    def test_sequence_never_goes_negative(self):
        """Positive multipliers preserve a non-negative policy score."""
        e = compute_sequence_energy_bound(["IL"] * 50, energy_initial=0.001, n_nodes=1)
        assert e >= 0.0

    def test_empty_sequence_returns_initial(self):
        e = compute_sequence_energy_bound([], 42.0, n_nodes=10)
        assert e == 42.0

    def test_proof_single_operator(self):
        proof = prove_sequence_lyapunov(["REMESH"])
        assert proof.is_net_contractive  # policy product = 1.0
        assert proof.cumulative_product == 1.0
