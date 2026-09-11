r"""Tests for the R6 controlled additive-character number theory.

The cyclic Goldbach convolution equals ``IFFT(hat 1_P^2)`` (additive reading is
Fourier), and the candidate TNFR phase observable shows no discriminating excess
over the classical power spectrum under density-matched controls — a constructive
negative keeping ``NT-P06`` OPEN.
"""

from __future__ import annotations

import numpy as np
import pytest

from tnfr.mathematics import additive_resonance as ar
from tnfr.mathematics.additive_resonance import (
    ExcessResult,
    constellation_indicator,
    cramer_indicator,
    cyclic_goldbach,
    discrimination_z,
    excess_over_fourier,
    goldbach_fourier_residual,
    matched_random_indicator,
    phase_curvature_energy,
    power_concentration,
    prime_indicator,
    shuffled_indicator,
)


def _reduction_tol(indicator: np.ndarray) -> float:
    # backward-error bound for the FFT path: eps * N * ||1||_2^2 (energy ~ count).
    count = float(indicator.sum())
    return np.finfo(float).eps * len(indicator) * max(count, 1.0) ** 2


# --------------------------------------------------------------------------- #
# Prime indicator and controls
# --------------------------------------------------------------------------- #
def test_prime_indicator_matches_known_primes():
    ind = prime_indicator(30)
    primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
    assert set(np.flatnonzero(ind).tolist()) == primes


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_prime_indicator_small_edges(n):
    ind = prime_indicator(n)
    assert len(ind) == max(n, 0)
    if n >= 3:
        assert ind[2] == 1.0


def test_matched_and_shuffle_preserve_count():
    primes = prime_indicator(512)
    count = int(primes.sum())
    assert int(matched_random_indicator(count, 512, seed=3).sum()) == count
    assert int(shuffled_indicator(primes, seed=3).sum()) == count


def test_cramer_density_is_prime_like():
    ind = cramer_indicator(2048, seed=5)
    # Cramer count should sit near pi(N); allow a wide band (random model)
    assert 250 <= int(ind.sum()) <= 380


def test_constellation_indicator_twin_starts():
    ind = constellation_indicator(30, (0, 2))
    twin_starts = {3, 5, 11, 17, 29}  # n with n and n+2 both prime, n < 30
    assert set(np.flatnonzero(ind).tolist()) == twin_starts


# --------------------------------------------------------------------------- #
# Required: the exact Fourier reduction (additive reading is Fourier)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [128, 256, 512])
def test_cyclic_goldbach_equals_spectral_square(n):
    ind = prime_indicator(n)
    assert goldbach_fourier_residual(ind) <= _reduction_tol(ind)


def test_cyclic_goldbach_counts_ordered_pairs():
    # small explicit check on Z/10Z: r_2(m) = #ordered prime pairs summing to m
    ind = prime_indicator(10)  # primes {2,3,5,7}
    r2 = cyclic_goldbach(ind)
    # m=5: (2,3),(3,2) -> 2 ; m=10%10=0: (3,7),(7,3),(5,5) -> 3
    assert r2[5] == 2
    assert r2[0] == 3


# --------------------------------------------------------------------------- #
# Required: relabel / phase-convention invariance
# --------------------------------------------------------------------------- #
def test_power_concentration_is_translation_invariant():
    primes = prime_indicator(512)
    base = power_concentration(primes)
    assert np.isclose(base, power_concentration(np.roll(primes, 23)))


def test_power_concentration_is_conjugation_invariant():
    # reversing n -> -n conjugates the spectrum; |hat 1|^2 is unchanged
    primes = prime_indicator(512)
    assert np.isclose(power_concentration(primes),
                      power_concentration(np.roll(primes[::-1], 1)))


def test_goldbach_is_translation_equivariant():
    primes = prime_indicator(256)
    rolled = cyclic_goldbach(np.roll(primes, 13))
    shifted = np.roll(cyclic_goldbach(primes), 26)
    assert np.allclose(rolled, shifted)


# --------------------------------------------------------------------------- #
# Required: Cramer / shuffle controls and the NT-P06 constructive negative
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", ["matched", "cramer", "shuffle"])
def test_tnfr_phase_does_not_exceed_fourier(kind):
    result = excess_over_fourier(1024, seed=0, n_controls=30, kind=kind)
    assert isinstance(result, ExcessResult)
    assert result.tnfr_exceeds is False


def test_classical_observable_detects_prime_structure():
    # sanity: the primes ARE structured under the classical (Fourier) observable
    result = excess_over_fourier(1024, seed=0, n_controls=30, kind="matched")
    assert abs(result.classical_z) > 5.0
    assert abs(result.tnfr_z) < abs(result.classical_z)


def test_prime_constellation_signature_is_classical():
    # twin-prime starts also stand out under the classical observable vs matched
    twins = constellation_indicator(2048, (0, 2))
    count = int(twins.sum())
    z = discrimination_z(power_concentration, twins, kind="matched",
                         n_controls=30, seed=1)
    assert abs(z) > 3.0
    assert count > 0


def test_excess_result_is_reproducible():
    a = excess_over_fourier(512, seed=7, n_controls=20, kind="matched")
    b = excess_over_fourier(512, seed=7, n_controls=20, kind="matched")
    assert a == b


def test_phase_curvature_energy_is_finite():
    for ind in (prime_indicator(256),
                matched_random_indicator(50, 256, seed=2)):
        assert np.isfinite(phase_curvature_energy(ind))


# --------------------------------------------------------------------------- #
# Guards and exports
# --------------------------------------------------------------------------- #
def test_unknown_control_kind_rejected():
    with pytest.raises(ValueError):
        discrimination_z(power_concentration, prime_indicator(128),
                         kind="bogus", n_controls=5, seed=0)


def test_module_exports_complete():
    expected = {
        "prime_indicator",
        "cramer_indicator",
        "shuffled_indicator",
        "matched_random_indicator",
        "constellation_indicator",
        "fourier_spectrum",
        "cyclic_goldbach",
        "goldbach_fourier_residual",
        "power_concentration",
        "phase_curvature_energy",
        "discrimination_z",
        "ExcessResult",
        "excess_over_fourier",
    }
    assert expected <= set(ar.__all__)
