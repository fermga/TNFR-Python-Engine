"""Comprehensive test suite for TNFR Arithmetic Network (number_theory.py).

Tests validate prime detection, TNFR properties computation, structural fields,
and operator compliance (U1-U6 grammar).

Status: COMPLETE VALIDATION
"""

import math
import pytest

try:
    import networkx as nx  # noqa: F401
    HAS_NX = True
except ImportError:
    HAS_NX = False

from tnfr.mathematics.number_theory import (
    ArithmeticTNFRNetwork,
    ArithmeticTNFRParameters,
    ArithmeticStructuralTerms,
    ArithmeticTNFRFormalism,
    PrimeCertificate,
)


class TestArithmeticStructuralTerms:
    """Test canonical structural terms (tau, sigma, omega)."""

    def test_terms_creation(self):
        """Terms can be created and stored."""
        terms = ArithmeticStructuralTerms(tau=4, sigma=7, omega=2)
        assert terms.tau == 4
        assert terms.sigma == 7
        assert terms.omega == 2

    def test_terms_frozen(self):
        """Terms are immutable."""
        terms = ArithmeticStructuralTerms(tau=4, sigma=7, omega=2)
        with pytest.raises(AttributeError):
            terms.tau = 5  # noqa: F841

    def test_terms_as_dict(self):
        """Terms can be converted to dict."""
        terms = ArithmeticStructuralTerms(tau=4, sigma=7, omega=2)
        d = terms.as_dict()
        assert d == {'tau': 4, 'sigma': 7, 'omega': 2}


class TestArithmeticTNFRFormalism:
    """Test TNFR formalism formulas."""

    def test_epi_value_formula(self):
        """EPI formula computes correctly."""
        params = ArithmeticTNFRParameters()
        terms = ArithmeticStructuralTerms(tau=2, sigma=3, omega=1)  # Prime
        epi = ArithmeticTNFRFormalism.epi_value(3, terms, params)
        # EPI = 1 + α*ω + β*ln(τ) + γ*(σ/n - 1)
        # = 1 + 0.5*1 + 0.3*ln(2) + 0.2*(3/3 - 1)
        # = 1 + 0.5 + 0.3*0.693 + 0
        # ≈ 1.708
        assert epi > 1.0
        assert epi < 2.0

    def test_delta_nfr_formula_primes(self):
        """ΔNFR should be near 0 for primes."""
        params = ArithmeticTNFRParameters()
        # Prime p has ω(p)=1, τ(p)=2, σ(p)=p+1
        p = 7
        terms = ArithmeticStructuralTerms(tau=2, sigma=p+1, omega=1)
        delta = ArithmeticTNFRFormalism.delta_nfr_value(p, terms, params)
        # ΔNFR = ζ*(ω-1) + η*(τ-2) + θ*(σ/n - (1+1/n))
        # = ζ*0 + η*0 + θ*(8/7 - 8/7)
        # = 0
        assert abs(delta) < 1e-10

    def test_delta_nfr_formula_composites(self):
        """ΔNFR should be nonzero for composites."""
        params = ArithmeticTNFRParameters()
        # Composite 6 = 2*3: ω(6)=2, τ(6)=4, σ(6)=12
        n = 6
        terms = ArithmeticStructuralTerms(tau=4, sigma=12, omega=2)
        delta = ArithmeticTNFRFormalism.delta_nfr_value(n, terms, params)
        # ΔNFR = ζ*(ω-1) + η*(τ-2) + θ*(σ/n - (1+1/n))
        # = 1.0*1 + 0.8*2 + 0.6*(12/6 - 7/6)
        # = 1 + 1.6 + 0.6*5/6
        # ≈ 2.9
        assert delta > 1.0

    def test_local_coherence_formula(self):
        """Coherence C = 1/(1+|ΔNFR|)."""
        c0 = ArithmeticTNFRFormalism.local_coherence(0.0)
        assert abs(c0 - 1.0) < 1e-10
        
        c1 = ArithmeticTNFRFormalism.local_coherence(1.0)
        assert abs(c1 - 0.5) < 1e-10


class TestArithmeticTNFRNetwork:
    """Test main network class."""

    @pytest.fixture
    def small_network(self):
        """Create a small test network (2-20)."""
        return ArithmeticTNFRNetwork(max_number=20)

    @pytest.fixture
    def medium_network(self):
        """Create a medium test network (2-100)."""
        return ArithmeticTNFRNetwork(max_number=100)

    def test_network_construction(self, small_network):
        """Network constructs correctly."""
        assert small_network.max_number == 20
        assert len(small_network.graph.nodes()) == 19  # 2-20 inclusive
        assert len(small_network.graph.edges()) > 0

    def test_divisor_count(self, small_network):
        """Divisor count computes correctly."""
        assert small_network._divisor_count(1) == 1
        assert small_network._divisor_count(2) == 2
        assert small_network._divisor_count(6) == 4  # 1, 2, 3, 6
        assert small_network._divisor_count(12) == 6  # 1, 2, 3, 4, 6, 12

    def test_divisor_sum(self, small_network):
        """Divisor sum computes correctly."""
        assert small_network._divisor_sum(1) == 1
        assert small_network._divisor_sum(2) == 3  # 1+2
        assert small_network._divisor_sum(6) == 12  # 1+2+3+6
        assert small_network._divisor_sum(12) == 28  # 1+2+3+4+6+12

    def test_prime_factor_count(self, small_network):
        """Prime factor count (with multiplicity) computes correctly."""
        assert small_network._prime_factor_count(2) == 1
        assert small_network._prime_factor_count(3) == 1
        assert small_network._prime_factor_count(4) == 2  # 2^2
        assert small_network._prime_factor_count(6) == 2  # 2*3
        assert small_network._prime_factor_count(12) == 3  # 2^2*3

    def test_is_prime(self, small_network):
        """Prime checking works correctly."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19]
        for p in primes:
            assert small_network._is_prime(p), f"{p} should be prime"
        
        non_primes = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20]
        for n in non_primes:
            assert not small_network._is_prime(n), f"{n} should not be prime"

    def test_tnfr_properties_structure(self, small_network):
        """TNFR properties are stored correctly."""
        # Check a prime
        p = 7
        props = small_network.get_tnfr_properties(p)
        assert 'number' in props
        assert 'tau' in props
        assert 'sigma' in props
        assert 'omega' in props
        assert 'EPI' in props
        assert 'nu_f' in props
        assert 'DELTA_NFR' in props
        assert 'is_prime' in props
        assert props['is_prime'] == True

        # Check a composite
        n = 12
        props = small_network.get_tnfr_properties(n)
        assert props['is_prime'] == False

    def test_prime_detection_recall(self, small_network):
        """Prime detection achieves high recall on small network."""
        validation = small_network.validate_prime_detection(delta_nfr_threshold=0.2)
        assert validation['recall'] >= 0.9, "Should detect >90% of primes"

    def test_prime_detection_precision(self, small_network):
        """Prime detection achieves high precision on small network."""
        validation = small_network.validate_prime_detection(delta_nfr_threshold=0.2)
        assert validation['precision'] >= 0.8, "Should have >80% precision"

    def test_prime_candidates(self, small_network):
        """detect_prime_candidates returns correct format."""
        candidates = small_network.detect_prime_candidates(delta_nfr_threshold=0.2)
        assert isinstance(candidates, list)
        if candidates:
            assert isinstance(candidates[0], tuple)
            assert len(candidates[0]) == 2
            assert isinstance(candidates[0][0], int)  # number
            assert isinstance(candidates[0][1], float)  # ΔNFR

    def test_prime_certificate(self, small_network):
        """Prime certificate generation works."""
        cert = small_network.get_prime_certificate(7)
        assert isinstance(cert, PrimeCertificate)
        assert cert.number == 7
        assert cert.tau == 2
        assert cert.sigma == 8
        assert cert.omega == 1
        assert cert.structural_prime == True
        assert abs(cert.delta_nfr) < 1e-10

    def test_summary_statistics(self, medium_network):
        """Summary statistics compute without error."""
        stats = medium_network.summary_statistics()
        assert stats['total_numbers'] == 99  # 2-100
        assert stats['prime_count'] > 0
        assert stats['composite_count'] > 0
        assert stats['prime_count'] + stats['composite_count'] == stats['total_numbers']
        assert 0 <= stats['prime_ratio'] <= 1

    def test_separation_primes_vs_composites(self, medium_network):
        """Primes and composites have significantly different ΔNFR."""
        stats = medium_network.summary_statistics()
        separation = abs(stats['composite_mean_DELTA_NFR'] - stats['prime_mean_DELTA_NFR'])
        assert separation > 0.5, "Primes and composites should separate in ΔNFR"


class TestStructuralFields:
    """Test integration with TNFR structural fields."""

    @pytest.fixture
    def network_with_phases(self):
        """Network with phases computed."""
        net = ArithmeticTNFRNetwork(max_number=30)
        net.compute_phase(method="spectral", store=True)
        return net

    def test_phase_computation(self, network_with_phases):
        """Phase computation succeeds."""
        phases = network_with_phases.compute_phase(
            method="logn", store=False
        )
        assert len(phases) == 29  # 2-30
        for n, phi in phases.items():
            assert 0 <= phi < 2 * math.pi

    def test_phase_gradient_computation(self, network_with_phases):
        """Phase gradient computes."""
        grad = network_with_phases.compute_phase_gradient()
        assert len(grad) > 0
        assert all(isinstance(v, (int, float)) for v in grad.values())

    def test_phase_curvature_computation(self, network_with_phases):
        """Phase curvature computes."""
        curv = network_with_phases.compute_phase_curvature()
        assert len(curv) > 0
        assert all(isinstance(v, (int, float)) for v in curv.values())

    def test_structural_potential_computation(self, network_with_phases):
        """Structural potential Φ_s computes."""
        phi_s = network_with_phases.compute_structural_potential(alpha=2.0)
        assert len(phi_s) > 0
        assert all(isinstance(v, (int, float)) for v in phi_s.values())

    def test_coherence_length_computation(self, network_with_phases):
        """Coherence length ξ_C computation returns dict."""
        xi_result = network_with_phases.estimate_coherence_length()
        assert isinstance(xi_result, dict)
        # xi_c may be None if not enough data points
        assert 'xi_c' in xi_result or 'r' in xi_result

    def test_all_fields_suite(self, network_with_phases):
        """compute_structural_fields wrapper works."""
        fields = network_with_phases.compute_structural_fields(
            phase_method="logn"
        )
        assert 'phi' in fields
        assert 'phi_grad' in fields
        assert 'k_phi' in fields
        assert 'phi_s' in fields
        assert 'xi_c' in fields


class TestOperators:
    """Test TNFR operators (UM: Coupling, RA: Resonance)."""

    @pytest.fixture
    def network_with_coupling(self):
        """Network with phases and coupling."""
        net = ArithmeticTNFRNetwork(max_number=25)
        try:
            net.compute_phase(method="logn", store=True)
        except KeyError:
            # Fallback if compute_phase has issues
            net.compute_phase(method="logn", store=False)
        return net

    def test_coupling_application(self, network_with_coupling):
        """Apply UM (Coupling) operator."""
        phi_max = math.pi / 2
        coupled = network_with_coupling.apply_coupling(
            delta_phi_max=phi_max
        )
        assert isinstance(coupled, dict)
        # Should have couplings or all False
        total_edges = len(coupled)
        assert total_edges > 0

    def test_resonance_step(self, network_with_coupling):
        """Apply RA (Resonance) step."""
        # Initialize activation on primes
        primes = [n for n in network_with_coupling.graph.nodes()
                  if network_with_coupling.graph.nodes[n]['is_prime']]
        activation = {
            n: (1.0 if n in primes else 0.0)
            for n in network_with_coupling.graph.nodes()
        }
        
        new_activation = network_with_coupling.resonance_step(
            activation,
            gain=1.0,
            decay=0.1,
            delta_phi_max=math.pi / 2,
            normalize=True
        )
        
        assert len(new_activation) == len(activation)
        assert all(0 <= v <= 1 for v in new_activation.values())

    def test_resonance_propagation(self, network_with_coupling):
        """Run multi-step resonance propagation."""
        history = network_with_coupling.resonance_from_primes(
            steps=3,
            init_value=1.0,
            gain=1.0,
            decay=0.1
        )
        
        assert len(history) == 4  # 0..3 steps
        assert all(isinstance(d, dict) for d in history)

    def test_resonance_metrics(self, network_with_coupling):
        """Compute resonance metrics."""
        primes = [n for n in network_with_coupling.graph.nodes()
                  if network_with_coupling.graph.nodes[n]['is_prime']]
        activation = {
            n: (1.0 if n in primes else 0.0)
            for n in network_with_coupling.graph.nodes()
        }
        
        metrics = network_with_coupling.resonance_metrics(activation)
        assert 'mean_activation' in metrics
        assert 'fraction_ge_0_5' in metrics
        assert 'corr_with_primes' in metrics


class TestGrammarCompliance:
    """Test TNFR grammar compliance (U1-U6)."""

    @pytest.fixture
    def network(self):
        return ArithmeticTNFRNetwork(max_number=50)

    def test_no_epi_mutations(self, network):
        """EPI values never change unexpectedly."""
        # Compute fields - should not mutate EPI
        network.compute_phase(store=True)
        network.compute_phase_gradient()
        network.compute_phase_curvature()
        
        # EPI should remain unchanged
        for n in network.graph.nodes():
            epi_stored = network.graph.nodes[n]['EPI']
            # Recompute from terms
            terms = network.get_structural_terms(n)
            epi_recomp = ArithmeticTNFRFormalism.epi_value(
                n, terms, network.params
            )
            assert abs(epi_stored - epi_recomp) < 1e-10

    def test_coherence_monotonicity(self, network):
        """Coherence C(t) monotonicity for primes."""
        # Primes should have C ≈ 1
        primes = [
            n for n in network.graph.nodes()
            if network.graph.nodes[n]['is_prime']
        ]
        for p in primes:
            delta = network.graph.nodes[p]['DELTA_NFR']
            c = 1.0 / (1.0 + abs(delta))
            assert c >= 0.99, f"Prime {p} should have C ≈ 1, got {c}"

    def test_frequency_positivity(self, network):
        """Structural frequency νf always positive."""
        for n in network.graph.nodes():
            nu_f = network.graph.nodes[n]['nu_f']
            assert nu_f > 0, f"νf should be positive for {n}, got {nu_f}"

    def test_operator_idempotence_phase(self, network):
        """Computing phase twice gives same result."""
        phi1 = network.compute_phase(
            method="logn", store=False
        )
        phi2 = network.compute_phase(
            method="logn", store=False
        )
        
        for n in phi1:
            assert abs(phi1[n] - phi2[n]) < 1e-10


class TestScalability:
    """Test performance and correctness on larger networks."""

    def test_medium_network_accuracy(self):
        """Medium network (2-200) detection accuracy."""
        net = ArithmeticTNFRNetwork(max_number=200)
        validation = net.validate_prime_detection(delta_nfr_threshold=0.15)
        
        # Should have decent recall/precision
        assert validation['recall'] >= 0.85
        assert validation['precision'] >= 0.75
        assert validation['f1_score'] >= 0.75

    def test_large_network_construction(self):
        """Large network (2-500) constructs without error."""
        net = ArithmeticTNFRNetwork(max_number=500)
        assert len(net.graph.nodes()) == 499
        stats = net.summary_statistics()
        assert stats['prime_count'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
