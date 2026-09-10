"""Tests for the upgraded Simple SDK — tetrad, balance, telemetry.

Validates that the advanced TNFR physics stack (Structural Field Tetrad,
scoped balance diagnostics, integrity monitoring, grammar-aware dynamics) is
correctly exposed through the simplified Network API.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tnfr.constants.aliases import ALIAS_DNFR
from tnfr.sdk.simple import (
    TNFR,
    ConservationReport,
    FactorizationReport,
    Network,
    NodalDynamicsReport,
    NodalStateReport,
    PrimalityReport,
    Results,
    TetradSnapshot,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_ring() -> Network:
    """5-node ring network — minimal connected topology."""
    return TNFR.create(5, name="ring5").ring()


@pytest.fixture
def medium_random() -> Network:
    """15-node random network — realistic density."""
    return TNFR.create(15, name="random15").random(0.3)


# ---------------------------------------------------------------------------
# TetradSnapshot dataclass
# ---------------------------------------------------------------------------


class TestTetradSnapshot:
    """TetradSnapshot creation and methods."""

    def test_empty_snapshot_summary(self):
        snap = TetradSnapshot()
        assert "empty" in snap.summary()

    def test_empty_snapshot_is_safe(self):
        snap = TetradSnapshot()
        safety = snap.is_safe()
        assert safety["overall"] is True

    def test_tetrad_from_network(self, small_ring: Network):
        snap = small_ring.tetrad()
        assert isinstance(snap, TetradSnapshot)
        assert len(snap.phi_s) == 5
        assert len(snap.grad_phi) == 5
        assert len(snap.k_phi) == 5
        assert isinstance(snap.xi_c, float)
        assert len(snap.j_phi) == 5
        assert len(snap.j_dnfr) == 5

    def test_tetrad_summary_nonempty(self, small_ring: Network):
        snap = small_ring.tetrad()
        summary = snap.summary()
        assert "Phi_s" in summary
        assert "N=5" in summary

    def test_tetrad_safety_returns_all_keys(self, small_ring: Network):
        safety = small_ring.tetrad().is_safe()
        for key in (
            "phi_s_safe",
            "grad_phi_safe",
            "k_phi_safe",
            "xi_c_safe",
            "overall",
        ):
            assert key in safety


# ---------------------------------------------------------------------------
# ConservationReport dataclass
# ---------------------------------------------------------------------------


class TestConservationReport:
    """ConservationReport creation and methods."""

    def test_default_report_is_unsampled(self):
        report = ConservationReport()
        assert report.lyapunov_stable is True
        assert report.candidate_energy_nonincreasing is None
        assert report.candidate_energy_within_numerical_tolerance is None
        assert report.candidate_energy_derivative is None
        assert report.balance_quality is None
        assert "UNSAMPLED" in report.summary()

    def test_exact_candidate_trend_is_separate_from_legacy_tolerance(self):
        report = ConservationReport(
            lyapunov_stable=True,
            lyapunov_derivative=1e-7,
            sample_available=True,
        )
        assert report.candidate_energy_nonincreasing is False
        assert report.candidate_energy_within_numerical_tolerance is True
        assert "INCREASING" in report.summary()

    def test_nonfinite_candidate_derivative_has_no_trend_verdict(self):
        report = ConservationReport(
            lyapunov_stable=False,
            lyapunov_derivative=float("nan"),
            sample_available=True,
        )
        assert report.candidate_energy_nonincreasing is None
        assert "UNDEFINED" in report.summary()

    def test_conservation_from_network(self, small_ring: Network):
        report = small_ring.conservation()
        assert isinstance(report, ConservationReport)
        assert isinstance(report.noether_charge, float)
        assert isinstance(report.energy, float)
        assert isinstance(report.lyapunov_stable, bool)
        assert report.structural_charge == report.noether_charge
        assert report.candidate_energy == report.energy
        assert report.sample_available is False


# ---------------------------------------------------------------------------
# Network.tetrad(), .fields(), .telemetry()
# ---------------------------------------------------------------------------


class TestNetworkFields:
    """Structural Field Tetrad integration in Network."""

    def test_tetrad_returns_correct_type(self, small_ring: Network):
        assert isinstance(small_ring.tetrad(), TetradSnapshot)

    def test_fields_returns_dict(self, small_ring: Network):
        f = small_ring.fields()
        assert isinstance(f, dict)
        assert "phi_s" in f
        assert "grad_phi" in f
        assert "k_phi" in f
        assert "xi_c" in f
        assert "j_phi" in f
        assert "j_dnfr" in f

    def test_telemetry_returns_dict(self, small_ring: Network):
        t = small_ring.telemetry()
        assert isinstance(t, dict)
        assert len(t) > 0

    def test_tensor_invariants(self, small_ring: Network):
        inv = small_ring.tensor_invariants()
        assert isinstance(inv, dict)
        assert "energy_density" in inv

    def test_emergent_fields(self, small_ring: Network):
        ef = small_ring.emergent_fields()
        assert isinstance(ef, dict)
        assert "chirality" in ef


# ---------------------------------------------------------------------------
# Network.conservation()
# ---------------------------------------------------------------------------


class TestNetworkConservation:
    """Finite structural-balance integration in Network."""

    def test_conservation_report_keys(self, small_ring: Network):
        c = small_ring.conservation()
        assert hasattr(c, "noether_charge")
        assert hasattr(c, "energy")
        assert hasattr(c, "lyapunov_stable")
        assert hasattr(c, "structural_charge")
        assert hasattr(c, "candidate_energy")
        assert hasattr(c, "balance_quality")

    def test_conservation_multiple_calls_track_snapshots(self, small_ring: Network):
        """Calling conservation() twice populates Lyapunov derivative."""
        c1 = small_ring.conservation()
        # Evolve to change state
        small_ring.evolve(1)
        c2 = small_ring.conservation()
        # Second call should have a real Lyapunov derivative
        assert c1.sample_available is False
        assert c2.sample_available is True
        assert isinstance(c2.lyapunov_derivative, float)
        assert isinstance(c2.candidate_energy_derivative, float)
        assert isinstance(c2.balance_quality, float)

    def test_balance_alerts_are_read_only_and_do_not_validate_grammar(
        self, small_ring: Network
    ):
        before = {
            node: dict(data) for node, data in small_ring.G.nodes(data=True)
        }
        baseline = small_ring.balance_alerts()
        after = {
            node: dict(data) for node, data in small_ring.G.nodes(data=True)
        }
        assert before == after
        assert baseline["sample_available"] is False
        assert baseline["grammar_validated"] is False
        assert baseline["violations_detected"] is False

    def test_legacy_grammar_violations_method_is_balance_alias(
        self, small_ring: Network
    ):
        result = small_ring.grammar_violations()
        assert result["grammar_validation_applicable"] is False
        assert result["grammar_validated"] is False
        assert result["violations_detected"] is False


# ---------------------------------------------------------------------------
# Network.evolve_grammar_aware()
# ---------------------------------------------------------------------------


class TestGrammarAwareDynamics:
    """Grammar-aware evolution preserves coherence."""

    def test_evolve_grammar_aware_returns_self(self, small_ring: Network):
        result = small_ring.evolve_grammar_aware(steps=2)
        assert result is small_ring

    def test_evolve_grammar_aware_maintains_coherence(self, small_ring: Network):
        c_before = small_ring.coherence()
        small_ring.evolve_grammar_aware(steps=3)
        c_after = small_ring.coherence()
        # Should not catastrophically break (allow some tolerance)
        assert c_after >= c_before * 0.5


# ---------------------------------------------------------------------------
# Network.integrity_check()
# ---------------------------------------------------------------------------


class TestIntegrityMonitor:
    """Structural integrity monitoring."""

    def test_integrity_check_returns_dict(self, small_ring: Network):
        report = small_ring.integrity_check()
        assert isinstance(report, dict)
        if report:  # non-empty if module available
            assert "operator" in report
            assert "nodes_checked" in report
            assert "pass_rate" in report


# ---------------------------------------------------------------------------
# Results — full_summary, to_dict, is_coherent, is_stable
# ---------------------------------------------------------------------------


class TestResults:
    """Upgraded Results with tetrad and conservation."""

    def test_results_has_tetrad(self, small_ring: Network):
        r = small_ring.results()
        assert isinstance(r, Results)
        assert r.tetrad is not None
        assert isinstance(r.tetrad, TetradSnapshot)

    def test_results_has_conservation(self, small_ring: Network):
        r = small_ring.results()
        assert r.conservation is not None
        assert isinstance(r.conservation, ConservationReport)

    def test_results_has_unified_fields(self, small_ring: Network):
        r = small_ring.results()
        assert r.unified_fields is not None
        assert isinstance(r.unified_fields, dict)

    def test_full_summary_multiline(self, small_ring: Network):
        r = small_ring.results()
        summary = r.full_summary()
        assert "\n" in summary
        assert "Tetrad" in summary

    def test_to_dict_serializable(self, small_ring: Network):
        r = small_ring.results()
        d = r.to_dict()
        assert isinstance(d, dict)
        assert "coherence" in d
        assert isinstance(d["coherence"], float)

    def test_is_coherent_and_is_stable(self, small_ring: Network):
        r = small_ring.results()
        assert isinstance(r.is_coherent(), bool)
        assert isinstance(r.is_stable(), bool)


# ---------------------------------------------------------------------------
# TNFR.analyze() — one-shot comprehensive analysis
# ---------------------------------------------------------------------------


class TestTNFRAnalyze:
    @pytest.fixture(autouse=True)
    def _materialize_pressure_channel(self, small_ring: Network) -> None:
        """TNFR analysis requires an observed pressure channel per node."""

        for node in small_ring.G:
            small_ring.G.nodes[node][ALIAS_DNFR[0]] = 0.0

    """TNFR.analyze() one-shot comprehensive analysis."""

    def test_analyze_returns_complete_dict(self, small_ring: Network):
        analysis = TNFR.analyze(small_ring)
        assert isinstance(analysis, dict)
        assert "coherence" in analysis
        assert "tetrad" in analysis
        assert "conservation" in analysis
        assert "tensor_invariants" in analysis
        assert "emergent_fields" in analysis
        assert "features" in analysis

    def test_analyze_tetrad_type(self, small_ring: Network):
        analysis = TNFR.analyze(small_ring)
        assert isinstance(analysis["tetrad"], TetradSnapshot)


# ---------------------------------------------------------------------------
# TNFR.compare() — updated comparison
# ---------------------------------------------------------------------------


class TestTNFRCompare:
    """Updated compare with conservation data."""

    def test_compare_includes_conservation(self):
        n1 = TNFR.create(5).ring()
        n2 = TNFR.create(8).ring()
        comp = TNFR.compare(n1, n2)
        assert comp["count"] == 2
        # Should have conservation data in results
        for r in comp["results"]:
            assert "coherence" in r


# ---------------------------------------------------------------------------
# Network.info() — features dict
# ---------------------------------------------------------------------------


class TestNetworkInfo:
    """Network.info() includes feature availability."""

    def test_info_has_features(self, small_ring: Network):
        info = small_ring.info()
        assert "features" in info
        for key in (
            "fields",
            "conservation",
            "integrity",
            "grammar_dynamics",
            "optimization",
        ):
            assert key in info["features"]
            assert isinstance(info["features"][key], bool)


# ---------------------------------------------------------------------------
# SDK __init__.py exports
# ---------------------------------------------------------------------------


class TestSDKExports:
    """Verify new types are importable from tnfr.sdk."""

    def test_import_tetrad_snapshot(self):
        from tnfr.sdk import TetradSnapshot as TS

        assert TS is TetradSnapshot

    def test_import_conservation_report(self):
        from tnfr.sdk import ConservationReport as CR

        assert CR is ConservationReport

    def test_import_factorization_report(self):
        from tnfr.sdk import FactorizationReport as FR

        assert FR is FactorizationReport

    def test_import_primality_report(self):
        from tnfr.sdk import PrimalityReport as PR

        assert PR is PrimalityReport

    def test_import_nodal_state_report(self):
        from tnfr.sdk import NodalStateReport as NSR

        assert NSR is NodalStateReport

    def test_import_nodal_dynamics_report(self):
        from tnfr.sdk import NodalDynamicsReport as NDR

        assert NDR is NodalDynamicsReport


class TestNodalDynamicsBridge:
    """SDK nodal-dynamics diagnostics for TNFR equation study."""

    @pytest.fixture(autouse=True)
    def _materialize_pressure_channel(self, small_ring: Network) -> None:
        """Make the diagnostic pressure explicit instead of assuming zero."""

        for node in small_ring.G:
            small_ring.G.nodes[node][ALIAS_DNFR[0]] = 0.0

    def test_nodal_state_returns_report(self, small_ring: Network):
        state = small_ring.nodal_state(0)
        assert isinstance(state, NodalStateReport)
        assert state.node == 0
        assert isinstance(state.expected_depi_dt, float)
        assert isinstance(state.d2epi_dt2, float)

    def test_nodal_scan_returns_report(self, small_ring: Network):
        report = small_ring.nodal_scan()
        assert isinstance(report, NodalDynamicsReport)
        assert len(report.nodes) == len(small_ring.G.nodes())
        top = report.top_pressure_nodes(3)
        assert len(top) <= 3

    def test_nodal_profile_returns_dict(self, small_ring: Network):
        profile = small_ring.nodal_profile(0)
        assert isinstance(profile, dict)
        assert "expected_depi_dt" in profile
        assert "delta_nfr" in profile

    def test_tnfr_analyze_includes_nodal_dynamics(self, small_ring: Network):
        analysis = TNFR.analyze(small_ring)
        assert "nodal_dynamics" in analysis
        assert isinstance(analysis["nodal_dynamics"], NodalDynamicsReport)


class TestSDKFactorizationBridge:
    """SDK bridge between canonical factorization and network telemetry."""

    def test_tnfr_factorize_returns_report(self):
        report = TNFR.factorize(91)
        assert isinstance(report, FactorizationReport)
        assert report.n == 91
        assert report.modulus > 0
        assert isinstance(report.candidate_factors, list)
        assert isinstance(report.telemetry, dict)
        assert "delta_nfr" in report.telemetry

    def test_network_factorize_includes_synergy(self, small_ring: Network):
        report = small_ring.factorize(91)
        assert isinstance(report, FactorizationReport)
        assert report.network_synergy is not None
        assert "synergy_index" in report.network_synergy
        assert "coherence_alignment" in report.network_synergy


class TestSDKPrimalityBridge:
    """SDK bridge between canonical primality module and network telemetry."""

    def test_tnfr_primality_returns_report(self):
        report = TNFR.primality(97)
        assert isinstance(report, PrimalityReport)
        assert report.n == 97
        assert report.is_prime is True
        assert abs(report.delta_nfr) <= report.tolerance
        assert isinstance(report.components, dict)
        assert isinstance(report.triad, dict)

    def test_report_summary_does_not_overclassify_direct_invalid_instance(self):
        report = PrimalityReport(
            n=1, is_prime=False, delta_nfr=float("inf"), tolerance=0.0
        )
        assert "status=not-prime" in report.summary()

    def test_network_primality_includes_synergy(self, small_ring: Network):
        report = small_ring.primality(91)
        assert isinstance(report, PrimalityReport)
        assert report.network_synergy is not None
        assert "synergy_index" in report.network_synergy
        assert "coherence_alignment" in report.network_synergy
        assert small_ring.is_prime(91) is False

    @pytest.mark.parametrize("candidate", [True, 17.5, float("nan")])
    def test_primality_rejects_non_integral_candidates(self, candidate):
        with pytest.raises((TypeError, ValueError)):
            TNFR.primality(candidate)

    @pytest.mark.parametrize("tolerance", [True, -1.0, float("nan"), float("inf")])
    def test_primality_rejects_invalid_tolerance(self, tolerance):
        with pytest.raises((TypeError, ValueError)):
            TNFR.primality(17, tolerance=tolerance)


class TestResearchFunctions:
    """New SDK research/teaching functions: trajectory, operators, explain."""

    def test_trajectory_records_per_operator(self, small_ring: Network):
        hist = small_ring.trajectory(cycles=2, sequence="basic_activation")
        assert len(hist) > 0
        assert len(hist) % 2 == 0  # two cycles
        for snap in hist:
            assert "step" in snap and "operator" in snap
            assert isinstance(snap["coherence"], float)
            assert isinstance(snap["sense_index"], float)
        assert [s["step"] for s in hist] == list(range(1, len(hist) + 1))

    def test_trajectory_unknown_sequence_raises(self, small_ring: Network):
        with pytest.raises(Exception):
            small_ring.trajectory(sequence="does_not_exist")

    def test_operators_catalog_has_13(self):
        catalog = TNFR.operators()
        assert isinstance(catalog, list)
        assert len(catalog) == 13
        names = {op["name"] for op in catalog}
        assert "Emission" in names
        assert "Mutation" in names

    def test_operators_single_emission(self):
        emi = TNFR.operators("emission")
        assert emi["name"] == "Emission"
        assert emi["glyph"] == "AL"
        assert emi["channel"] == "EPI"
        assert "generator" in emi["roles"]

    def test_operators_accepts_glyph(self):
        il = TNFR.operators("IL")
        assert il["name"] == "Coherence"
        assert "stabilizer" in il["roles"]

    def test_explain_sequence_valid(self):
        info = TNFR.explain_sequence(["emission", "coherence", "silence"])
        assert info["valid"] is True
        assert info["starts_with_generator"] is True
        assert info["ends_with_closure"] is True
        assert len(info["roles"]) == 3

    def test_explain_sequence_invalid_no_generator(self):
        info = TNFR.explain_sequence(["coherence", "resonance"])
        assert info["valid"] is False
        assert info["starts_with_generator"] is False

    def test_explain_sequence_accepts_glyphs(self):
        info = TNFR.explain_sequence(["AL", "OZ", "IL", "SHA"])
        assert info["operators"][0] == "Emission"
        assert info["has_destabilizer"] is True
        assert info["has_stabilizer"] is True


class TestEmergentOntologyAndNumberTheory:
    """Emergent ontology + number theory functions (unified dNFR=0 template)."""

    def test_primes_matches_known(self):
        result = TNFR.primes(30)
        assert result["primes"] == [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        assert result["count"] == 10
        assert result["max_number"] == 30

    def test_magic_numbers_noble_gases(self):
        magic = TNFR.magic_numbers()
        assert magic[:6] == [2, 10, 18, 36, 54, 86]

    def test_element_noble_gas_zero_dnfr(self):
        neon = TNFR.element(10)
        assert neon["closed_shell"] is True
        assert neon["reactivity"] == 0.0
        assert neon["delta_nfr"] == 0.0

    def test_element_reactive_nonzero_dnfr(self):
        sodium = TNFR.element(11)
        assert sodium["closed_shell"] is False
        assert sodium["reactivity"] > 0.0

    def test_network_particle(self, small_ring: Network):
        winding = small_ring.winding()
        assert "winding" in winding
        assert "winding_class" in winding
        assert isinstance(winding["orientation_sign"], int)
        assert winding["telemetry_scope"] == "whole_graph_snapshot"
        assert small_ring.particle() == winding

    @pytest.mark.parametrize("value", [True, 10.9, float("nan")])
    def test_count_apis_reject_lossy_or_nonfinite_values(self, value):
        with pytest.raises((TypeError, ValueError)):
            TNFR.primes(value)
        with pytest.raises((TypeError, ValueError)):
            TNFR.magic_numbers(value)
        with pytest.raises((TypeError, ValueError)):
            TNFR.element(value)

    def test_network_phase(self, small_ring: Network):
        ph = small_ring.phase()
        assert ph["phase"] in ("non_life", "critical", "life")
        assert isinstance(ph["is_life"], bool)
        assert "order_parameter" in ph

    def test_network_gauge(self):
        net = TNFR.create(8, seed=3).ring().evolve(4)
        g = net.gauge()
        assert "dominant_regime" in g
        assert isinstance(g["regime_distribution"], dict)

    def test_network_spectrum(self):
        net = TNFR.create(8, seed=3).ring().evolve(3)
        s = net.spectrum()
        assert s["spectral_gap"] >= 0.0
        assert len(s["relaxation_rates"]) >= 1
        assert s["structural_rank"] >= 1

    def test_spectrum_uses_heterogeneous_nodal_frequencies(self):
        import networkx as nx

        graph = nx.path_graph(3)
        nx.set_node_attributes(graph, {0: 1.0, 1: 3.0, 2: 5.0}, "nu_f")
        spectrum = Network(graph).spectrum()
        assert spectrum["relaxation_rates"] == pytest.approx([0.0, 2.0, 7.0], abs=1e-12)
        assert spectrum["spectral_gap"] == pytest.approx(2.0)
        assert spectrum["coherence_length"] == pytest.approx(1.0)

    def test_disconnected_spectrum_does_not_skip_extra_stationary_mode(self):
        import networkx as nx

        graph = nx.Graph([(0, 1), (2, 3)])
        nx.set_node_attributes(graph, 1.0, "nu_f")
        spectrum = Network(graph).spectrum()
        assert spectrum["spectral_gap"] == 0.0
        assert spectrum["coherence_length"] == float("inf")

    @pytest.mark.parametrize("nu_f", [0.0, 0.25, 1.0, 4.0])
    def test_spectrum_coherence_length_is_independent_of_clock(self, nu_f):
        import math

        from tnfr.alias import set_attr
        from tnfr.constants.aliases import ALIAS_VF

        net = TNFR.create(7, seed=3).ring()
        for node in net.G:
            set_attr(net.G.nodes[node], ALIAS_VF, nu_f)

        spectrum = net.spectrum()
        # On C_7, lambda_2 = 1 - cos(2*pi/7). Changing reorganization
        # capacity rescales relaxation rates but preserves spatial geometry.
        geometry_gap = 1.0 - math.cos(2.0 * math.pi / 7.0)
        assert spectrum["spectral_gap"] == pytest.approx(nu_f * geometry_gap)
        assert spectrum["coherence_length"] == pytest.approx(
            1.0 / math.sqrt(geometry_gap)
        )

    def test_symbolic_layer_reads_shared_scalar_predicate(self):
        """The shell distance reuses only the scalar equilibrium predicate."""
        from tnfr.metrics.common import is_structural_equilibrium

        assert is_structural_equilibrium(TNFR.element(10)["delta_nfr"])  # Ne
        assert not is_structural_equilibrium(TNFR.element(11)["delta_nfr"])  # Na


class TestStructuralEquilibriumPrimitive:
    """Shared scalar kernel and predicate across distinct domain models."""

    def test_structural_coherence_unity_at_equilibrium(self):
        from tnfr.metrics.common import structural_coherence

        assert structural_coherence(0.0) == 1.0
        assert structural_coherence(0.0, 0.0) == 1.0

    def test_structural_coherence_monotone(self):
        from tnfr.metrics.common import structural_coherence

        assert (
            structural_coherence(0.0)
            > structural_coherence(1.0)
            > structural_coherence(5.0)
        )

    def test_structural_coherence_includes_depi(self):
        from tnfr.metrics.common import structural_coherence

        assert structural_coherence(1.0, 1.0) == 1.0 / 3.0

    def test_structural_coherence_avoids_finite_input_overflow(self):
        from tnfr.metrics.common import structural_coherence

        result = structural_coherence(1.0e308, 1.0e308)
        assert math.isfinite(result)
        assert result == pytest.approx(5.0e-309, rel=1.0e-12)

    def test_structural_coherence_preserves_vectorized_overflow_safety(self):
        from tnfr.metrics.common import structural_coherence

        result = structural_coherence(
            np.array([0.0, 1.0e308]), np.array([0.0, 1.0e308])
        )

        assert np.all(np.isfinite(result))
        np.testing.assert_allclose(result, np.array([1.0, 5.0e-309]), rtol=1.0e-12)

    def test_is_structural_equilibrium_default_tolerance(self):
        from tnfr.metrics.common import is_structural_equilibrium

        assert is_structural_equilibrium(0.0)
        assert is_structural_equilibrium(1e-4)  # below 1e-3 default
        assert not is_structural_equilibrium(0.5)

    def test_is_structural_equilibrium_custom_eps(self):
        from tnfr.metrics.common import is_structural_equilibrium

        assert is_structural_equilibrium(1e-13, eps_dnfr=1e-12)
        assert not is_structural_equilibrium(1e-10, eps_dnfr=1e-12)

    @pytest.mark.parametrize("value", [True, False, float("inf"), float("nan")])
    def test_structural_coherence_rejects_invalid_scalars(self, value):
        from tnfr.metrics.common import structural_coherence

        error = TypeError if isinstance(value, bool) else ValueError
        with pytest.raises(error):
            structural_coherence(value)

    def test_equilibrium_rejects_negative_tolerance(self):
        from tnfr.metrics.common import is_structural_equilibrium

        with pytest.raises(ValueError, match="non-negative"):
            is_structural_equilibrium(0.0, eps_dnfr=-1.0)

    def test_compute_coherence_uses_kernel(self):
        """compute_coherence delegates to the structural_coherence kernel."""
        from tnfr.metrics.common import compute_coherence, structural_coherence

        net = TNFR.create(6, seed=1).ring().evolve(3)
        c, dnfr_mean, depi_mean = compute_coherence(net.G, return_means=True)
        assert c == structural_coherence(dnfr_mean, depi_mean)

    def test_compute_coherence_avoids_intermediate_mean_overflow(self):
        import networkx as nx

        from tnfr.metrics.common import compute_coherence, structural_coherence

        graph = nx.path_graph(2)
        for node in graph:
            graph.nodes[node].update(delta_nfr=1.0e308, dEPI_dt=1.0e308)

        coherence, dnfr_mean, depi_mean = compute_coherence(
            graph, return_means=True
        )
        assert dnfr_mean == 1.0e308
        assert depi_mean == 1.0e308
        assert coherence == structural_coherence(dnfr_mean, depi_mean)
        assert coherence > 0.0

    def test_number_theory_local_coherence_delegates(self):
        """Arithmetic local coherence routes through the canonical kernel."""
        from tnfr.mathematics.number_theory import ArithmeticTNFRFormalism as F
        from tnfr.metrics.common import structural_coherence

        assert F.local_coherence(2.0) == structural_coherence(2.0)
        assert F.local_coherence(0.0) == 1.0


class TestFractalResonantNode:
    """NFR (Nodo Fractal Resonante) read-out: nodal topology + facets.

    Per TNFR.pdf section 1.4.1 the NFR has a nodal topology (radial/annular/
    multinodal) read from the canonical structural-potential geometry.
    """

    def test_classify_radial(self):
        import networkx as nx

        from tnfr.physics.fields import classify_nodal_topology

        r = classify_nodal_topology(nx.star_graph(9))
        assert r["topology"] == "radial"
        assert len(r["centers"]) == 1

    def test_classify_annular(self):
        import networkx as nx

        from tnfr.physics.fields import classify_nodal_topology

        assert classify_nodal_topology(nx.cycle_graph(10))["topology"] == "annular"
        assert classify_nodal_topology(nx.complete_graph(6))["topology"] == "annular"

    def test_classify_multinodal(self):
        import networkx as nx

        from tnfr.physics.fields import classify_nodal_topology

        r = classify_nodal_topology(nx.barbell_graph(5, 0))
        assert r["topology"] == "multinodal"
        assert len(r["centers"]) >= 2

    def test_network_nfr_ring_is_annular(self):
        net = TNFR.create(10, seed=2).ring().evolve(3)
        d = net.nfr()
        assert d["topology"] == "annular"
        assert set(d) >= {
            "topology",
            "centers",
            "concentration",
            "coherence",
            "zero_pressure_fraction",
            "equilibrium_fraction",
            "pressure_telemetry_available",
            "dynamic_telemetry_available",
            "depi_dt_source",
            "triad_available",
            "coherence_length",
            "triad",
            "n_nodes",
        }
        assert 0.0 <= d["equilibrium_fraction"] <= 1.0
        assert d["pressure_telemetry_available"] is True
        assert d["dynamic_telemetry_available"] is True
        assert d["depi_dt_source"] in {"recorded", "nodal_equation"}
        assert d["triad_available"] is True
        assert set(d["triad"]) == {"epi_mean", "vf_mean", "phase_sync"}

    def test_network_nfr_uniform_is_one_nfr(self):
        """The evolved fixture reaches the measured equilibrium tolerance."""
        net = TNFR.create(8, seed=1).ring().evolve(6)
        d = net.nfr()
        assert d["equilibrium_fraction"] == 1.0
        assert d["coherence"] >= 0.7

    def test_bare_graph_reports_missing_nodal_telemetry(self):
        import networkx as nx

        result = Network(nx.path_graph(3)).nfr()
        assert result["pressure_telemetry_available"] is False
        assert result["dynamic_telemetry_available"] is False
        assert result["depi_dt_source"] is None
        assert result["triad_available"] is False
        assert result["zero_pressure_fraction"] is None
        assert result["equilibrium_fraction"] is None
        assert result["coherence"] is None

    def test_nodal_state_exposes_coherence_facet(self):
        """The per-node micro-NFR exposes its constitutive coherence."""
        net = TNFR.create(6, seed=3).ring().evolve(4)
        n0 = list(net.G.nodes())[0]
        s = net.nodal_state(n0)
        assert hasattr(s, "coherence")
        assert 0.0 < s.coherence <= 1.0
        assert "coherence" in s.to_dict()

    def test_micro_and_macro_nfr_equilibrium_agree(self):
        """nodal_state (micro-NFR) and Network.nfr (macro-NFR) share the
        canonical equilibrium predicate and tolerance."""
        net = TNFR.create(8, seed=1).ring().evolve(6)
        scan = net.nodal_scan()
        micro = sum(1 for r in scan.nodes.values() if r.equilibrium) / len(scan.nodes)
        assert micro == net.nfr()["equilibrium_fraction"]
        assert scan.equilibrium_tolerance == pytest.approx(1e-3)

    def test_pulse_trajectory_records_rhythm_in_motion(self):
        """The pulse in motion: the rhythm forms as the NFR pulses resonate.

        Snapshots are time-blind; from a perturbed state the trajectory
        records the synchronization R(t) and the local-before-global cascade,
        and the collective pulse is computed once (evolution-invariant)."""
        import random

        from tnfr.alias import set_attr
        from tnfr.constants.aliases import ALIAS_THETA, ALIAS_VF

        net = TNFR.create(24, seed=0).ring()
        rng = random.Random(0)
        for n in net.G.nodes():
            set_attr(net.G.nodes[n], ALIAS_VF, rng.uniform(0.6, 1.4))
            set_attr(net.G.nodes[n], ALIAS_THETA, rng.uniform(0.0, 6.283))
        traj = net.pulse_trajectory(steps=8)
        assert traj["steps"] == 8
        for key in ("phase_coherence", "coherence", "local_resonance"):
            assert len(traj[key]) == 8
        assert all(0.0 <= r <= 1.0 for r in traj["phase_coherence"])
        assert all(0.0 <= x <= 1.0 for x in traj["local_resonance"])
        # the per-NFR pulses synchronize over the run (R rises)
        assert traj["synchronizing"] is True
        assert traj["delta_R"] > 0.0
        # local-before-global: per-NFR pulses lock with neighbours first, so
        # local resonance leads the global rhythm (robust margin)
        assert traj["local_resonance"][-1] > traj["phase_coherence"][-1]
        assert traj["local_leads_global"] is True
        assert isinstance(traj["collective_pulse"]["fundamental"], float)

    def test_pulse_trajectory_is_non_destructive(self):
        """The trajectory evolves a copy; the caller's network is untouched."""
        net = TNFR.create(12, seed=1).ring().evolve(2)
        before = net.coherence()
        histories_before = {
            node: tuple(data.get("glyph_history", ()))
            for node, data in net.G.nodes(data=True)
        }
        first = net.pulse_trajectory(steps=4)
        second = net.pulse_trajectory(steps=4)
        # identical first sample => the network was not advanced in place
        assert first["phase_coherence"][0] == pytest.approx(
            second["phase_coherence"][0]
        )
        assert net.coherence() == pytest.approx(before)
        assert {
            node: tuple(data.get("glyph_history", ()))
            for node, data in net.G.nodes(data=True)
        } == histories_before

    def test_pulse_trajectory_detaches_bound_pressure_callback_owner(self):
        """The probe must not mutate state owned by the caller's callback."""
        import threading

        class PressureRefresh:
            def __init__(self):
                self.calls = 0
                self.lock = threading.Lock()

            def refresh(self, graph):
                self.calls += 1
                for node in graph:
                    graph.nodes[node]["delta_nfr"] = 0.0

        net = TNFR.create(3, seed=1).ring()
        refresh = PressureRefresh()
        net.G.graph["compute_delta_nfr"] = refresh.refresh

        net.pulse_trajectory(steps=2)

        assert refresh.calls == 0

    def test_evolve_record_populates_rhythm_history(self):
        """evolve(record=True) records the canonical pulse-in-motion series.

        The engine's own metrics step samples kuramoto_R / C_steps after each
        cycle, surfaced by history() -- the resonance forming, not a snapshot.
        """
        import random

        from tnfr.alias import set_attr
        from tnfr.constants.aliases import ALIAS_THETA, ALIAS_VF

        net = TNFR.create(24, seed=0).ring()
        rng = random.Random(0)
        for n in net.G.nodes():
            set_attr(net.G.nodes[n], ALIAS_VF, rng.uniform(0.6, 1.4))
            set_attr(net.G.nodes[n], ALIAS_THETA, rng.uniform(0.0, 6.283))
        net.evolve(5, record=True)
        hist = net.history()
        assert len(hist["kuramoto_R"]) == 5
        assert len(hist["C_steps"]) == 5
        assert all(0.0 <= r <= 1.0 for r in hist["kuramoto_R"])
        # the per-NFR pulses synchronize over the run (the rhythm forms)
        assert hist["kuramoto_R"][-1] > hist["kuramoto_R"][0]

    def test_evolve_without_record_keeps_fast_path(self):
        """The default path records no per-step rhythm series."""
        net = TNFR.create(8, seed=1).ring().evolve(3)
        assert net.history()["kuramoto_R"] == []
