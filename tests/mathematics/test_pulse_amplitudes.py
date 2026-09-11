r"""Tests for pointed-pulse amplitudes on normal circulants (R2, N12).

The pulse ``h(t) = Σ_λ a_λ e^{-tλ}`` on the k-th power residue circulant has
amplitudes ``a_λ = m_λ/p`` (normalized spectral multiplicities): the pointed seed
``e₀`` excites every Fourier mode with equal weight ``1/p``.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd

from tnfr.mathematics.arithmetic_pulse import (
    cyclotomic_rank,
    pointed_pulse_hankel_rank,
)
from tnfr.mathematics import pulse_amplitudes as pa
from tnfr.mathematics.pulse_amplitudes import (
    amplitudes_basis_invariance_residual,
    certify_pulse_amplitudes,
    moment_reconstruction_residual,
    projection_amplitudes,
    pulse_amplitudes,
)

CASES = [(5, 2), (5, 3), (5, 4), (7, 2), (7, 3), (11, 2), (13, 3), (13, 4)]


# --------------------------------------------------------------------------- #
# The theorem: amplitude = multiplicity / n
# --------------------------------------------------------------------------- #
def test_pointed_circulant_amplitude_equals_multiplicity_over_n():
    for p, k in CASES:
        amps = pulse_amplitudes(p, k)
        proj = projection_amplitudes(p, k)
        for (_, m, a), (_, pamp) in zip(amps, proj):
            assert a == Fraction(m, p)              # exact amplitude
            assert abs(pamp - float(a)) < 1e-9      # spectral-projection check


def test_amplitudes_sum_to_one():
    for p, k in CASES:
        assert sum(a for _, _, a in pulse_amplitudes(p, k)) == Fraction(1)


# --------------------------------------------------------------------------- #
# Exact rational reconstruction of the moments
# --------------------------------------------------------------------------- #
def test_pulse_reconstruction_exact_rational():
    for p, k in CASES:
        assert moment_reconstruction_residual(p, k) < 1e-9


def test_complex_conjugate_pulse_is_real_when_expected():
    # conjugate eigenvalues carry equal (real) amplitudes, so the pulse is real
    for p, k in CASES:
        amps = pulse_amplitudes(p, k)
        for lam, m, a in amps:
            if abs(lam.imag) > 1e-9:
                mate = [(m2, a2) for lam2, m2, a2 in amps
                        if abs(lam2 - lam.conjugate()) < 1e-9]
                assert mate and mate[0] == (m, a)   # conjugate has equal weight


# --------------------------------------------------------------------------- #
# Basis invariance and the R2 rank link
# --------------------------------------------------------------------------- #
def test_amplitudes_basis_invariant():
    for p, k in CASES:
        assert amplitudes_basis_invariance_residual(p, k) < 1e-9


def test_hankel_order_unchanged_by_grouping():
    # #distinct amplitudes == R2 pulse rank == gcd(k, p-1) + 1
    for p, k in CASES:
        n = len(pulse_amplitudes(p, k))
        assert n == cyclotomic_rank(p, k) == gcd(k, p - 1) + 1
        assert n == pointed_pulse_hankel_rank(p, k)


# --------------------------------------------------------------------------- #
# Certificate + exports
# --------------------------------------------------------------------------- #
def test_certificate_bundle():
    for p, k in [(7, 3), (11, 2)]:
        c = certify_pulse_amplitudes(p, k)
        assert c.rank_matches_cyclotomy
        assert c.amplitudes_sum_to_one
        assert c.amplitude_equals_multiplicity
        assert c.moment_reconstruction_residual < 1e-9
        assert c.basis_invariance_residual < 1e-9
        assert "NT-P02b" in c.claim_status


def test_module_exports_complete():
    expected = {
        "circulant_eigenvalues", "fourier_basis", "pulse_amplitudes",
        "projection_amplitudes", "moment_reconstruction_residual",
        "amplitudes_basis_invariance_residual", "PulseAmplitudeCertificate",
        "certify_pulse_amplitudes",
    }
    assert expected <= set(pa.__all__)
