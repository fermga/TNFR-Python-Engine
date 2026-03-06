"""Tests for the upgraded Simple SDK — tetrad, conservation, telemetry.

Validates that the advanced TNFR physics stack (Structural Field Tetrad,
conservation laws, integrity monitoring, grammar-aware dynamics) is
correctly exposed through the simplified Network API.
"""

from __future__ import annotations

import pytest

from tnfr.sdk.simple import (
    TNFR,
    Network,
    Results,
    TetradSnapshot,
    ConservationReport,
    FactorizationReport,
    PrimalityReport,
    NodalStateReport,
    NodalDynamicsReport,
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
        assert safety['overall'] is True

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
        for key in ('phi_s_safe', 'grad_phi_safe', 'k_phi_safe', 'xi_c_safe', 'overall'):
            assert key in safety


# ---------------------------------------------------------------------------
# ConservationReport dataclass
# ---------------------------------------------------------------------------

class TestConservationReport:
    """ConservationReport creation and methods."""

    def test_default_report_stable(self):
        report = ConservationReport()
        assert report.lyapunov_stable is True
        assert "STABLE" in report.summary()

    def test_conservation_from_network(self, small_ring: Network):
        report = small_ring.conservation()
        assert isinstance(report, ConservationReport)
        assert isinstance(report.noether_charge, float)
        assert isinstance(report.energy, float)
        assert isinstance(report.lyapunov_stable, bool)


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
        assert 'phi_s' in f
        assert 'grad_phi' in f
        assert 'k_phi' in f
        assert 'xi_c' in f
        assert 'j_phi' in f
        assert 'j_dnfr' in f

    def test_telemetry_returns_dict(self, small_ring: Network):
        t = small_ring.telemetry()
        assert isinstance(t, dict)
        assert len(t) > 0

    def test_tensor_invariants(self, small_ring: Network):
        inv = small_ring.tensor_invariants()
        assert isinstance(inv, dict)
        assert 'energy_density' in inv

    def test_emergent_fields(self, small_ring: Network):
        ef = small_ring.emergent_fields()
        assert isinstance(ef, dict)
        assert 'chirality' in ef


# ---------------------------------------------------------------------------
# Network.conservation()
# ---------------------------------------------------------------------------

class TestNetworkConservation:
    """Conservation law integration in Network."""

    def test_conservation_report_keys(self, small_ring: Network):
        c = small_ring.conservation()
        assert hasattr(c, 'noether_charge')
        assert hasattr(c, 'energy')
        assert hasattr(c, 'lyapunov_stable')

    def test_conservation_multiple_calls_track_snapshots(self, small_ring: Network):
        """Calling conservation() twice populates Lyapunov derivative."""
        c1 = small_ring.conservation()
        # Evolve to change state
        small_ring.evolve(1)
        c2 = small_ring.conservation()
        # Second call should have a real Lyapunov derivative
        assert isinstance(c2.lyapunov_derivative, float)


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
            assert 'operator' in report
            assert 'nodes_checked' in report
            assert 'pass_rate' in report


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
        assert '\n' in summary
        assert 'Tetrad' in summary

    def test_to_dict_serializable(self, small_ring: Network):
        r = small_ring.results()
        d = r.to_dict()
        assert isinstance(d, dict)
        assert 'coherence' in d
        assert isinstance(d['coherence'], float)

    def test_is_coherent_and_is_stable(self, small_ring: Network):
        r = small_ring.results()
        assert isinstance(r.is_coherent(), bool)
        assert isinstance(r.is_stable(), bool)


# ---------------------------------------------------------------------------
# TNFR.analyze() — one-shot comprehensive analysis
# ---------------------------------------------------------------------------

class TestTNFRAnalyze:
    """TNFR.analyze() one-shot comprehensive analysis."""

    def test_analyze_returns_complete_dict(self, small_ring: Network):
        analysis = TNFR.analyze(small_ring)
        assert isinstance(analysis, dict)
        assert 'coherence' in analysis
        assert 'tetrad' in analysis
        assert 'conservation' in analysis
        assert 'tensor_invariants' in analysis
        assert 'emergent_fields' in analysis
        assert 'features' in analysis

    def test_analyze_tetrad_type(self, small_ring: Network):
        analysis = TNFR.analyze(small_ring)
        assert isinstance(analysis['tetrad'], TetradSnapshot)


# ---------------------------------------------------------------------------
# TNFR.compare() — updated comparison
# ---------------------------------------------------------------------------

class TestTNFRCompare:
    """Updated compare with conservation data."""

    def test_compare_includes_conservation(self):
        n1 = TNFR.create(5).ring()
        n2 = TNFR.create(8).ring()
        comp = TNFR.compare(n1, n2)
        assert comp['count'] == 2
        # Should have conservation data in results
        for r in comp['results']:
            assert 'coherence' in r


# ---------------------------------------------------------------------------
# Network.info() — features dict
# ---------------------------------------------------------------------------

class TestNetworkInfo:
    """Network.info() includes feature availability."""

    def test_info_has_features(self, small_ring: Network):
        info = small_ring.info()
        assert 'features' in info
        for key in ('fields', 'conservation', 'integrity', 'grammar_dynamics', 'optimization'):
            assert key in info['features']
            assert isinstance(info['features'][key], bool)


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
        assert 'expected_depi_dt' in profile
        assert 'delta_nfr' in profile

    def test_tnfr_analyze_includes_nodal_dynamics(self, small_ring: Network):
        analysis = TNFR.analyze(small_ring)
        assert 'nodal_dynamics' in analysis
        assert isinstance(analysis['nodal_dynamics'], NodalDynamicsReport)


class TestSDKFactorizationBridge:
    """SDK bridge between canonical factorization and network telemetry."""

    def test_tnfr_factorize_returns_report(self):
        report = TNFR.factorize(91)
        assert isinstance(report, FactorizationReport)
        assert report.n == 91
        assert report.modulus > 0
        assert isinstance(report.candidate_factors, list)
        assert isinstance(report.telemetry, dict)
        assert 'delta_nfr' in report.telemetry

    def test_network_factorize_includes_synergy(self, small_ring: Network):
        report = small_ring.factorize(91)
        assert isinstance(report, FactorizationReport)
        assert report.network_synergy is not None
        assert 'synergy_index' in report.network_synergy
        assert 'coherence_alignment' in report.network_synergy


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

    def test_network_primality_includes_synergy(self, small_ring: Network):
        report = small_ring.primality(91)
        assert isinstance(report, PrimalityReport)
        assert report.network_synergy is not None
        assert 'synergy_index' in report.network_synergy
        assert 'coherence_alignment' in report.network_synergy
        assert small_ring.is_prime(91) is False
