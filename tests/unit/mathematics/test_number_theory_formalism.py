"""Tests for the arithmetic TNFR formalism helper classes."""

from __future__ import annotations

import math

import pytest

from tnfr.mathematics import (
    ArithmeticTNFRNetwork,
    ArithmeticTNFRFormalism,
    ArithmeticStructuralTerms,
    PrimeCertificate,
)


@pytest.fixture(scope="module")
def small_network() -> ArithmeticTNFRNetwork:
    """Build a small network that still exercises enough composites/primes."""
    return ArithmeticTNFRNetwork(max_number=50)


def test_structural_terms_match_classical_invariants(small_network: ArithmeticTNFRNetwork) -> None:
    terms = small_network.get_structural_terms(12)
    assert isinstance(terms, ArithmeticStructuralTerms)
    assert terms.tau == 6  # divisors: 1,2,3,4,6,12
    assert terms.sigma == 28  # divisor sum
    assert terms.omega == 3  # 12 = 2^2 * 3
    assert terms.as_dict() == {'tau': 6, 'sigma': 28, 'omega': 3}


def test_prime_certificate_detects_structural_attractor(small_network: ArithmeticTNFRNetwork) -> None:
    prime_cert = small_network.get_prime_certificate(13)
    assert isinstance(prime_cert, PrimeCertificate)
    assert prime_cert.structural_prime
    assert math.isclose(prime_cert.delta_nfr, 0.0, abs_tol=prime_cert.tolerance)
    assert prime_cert.components is not None
    assert set(prime_cert.components.keys()) == {
        'factorization_pressure',
        'divisor_pressure',
        'sigma_pressure',
    }
    manual_components = ArithmeticTNFRFormalism.component_breakdown(
        13,
        small_network.get_structural_terms(13),
        small_network.params,
    )
    assert prime_cert.components == manual_components

    composite_cert = small_network.get_prime_certificate(12)
    assert not composite_cert.structural_prime
    assert abs(composite_cert.delta_nfr) > composite_cert.tolerance


def test_detect_prime_candidates_can_return_certificates(small_network: ArithmeticTNFRNetwork) -> None:
    certificates = small_network.detect_prime_candidates(
        delta_nfr_threshold=1e-9,
        tolerance=1e-12,
        return_certificates=True,
    )
    assert certificates, "Expected at least one candidate"
    assert all(isinstance(cert, PrimeCertificate) for cert in certificates)
    numbers = [cert.number for cert in certificates]
    assert numbers == sorted(numbers)
    assert all(small_network.graph.nodes[n]['is_prime'] for n in numbers)