"""Tests for the canonical arithmetic residue-network API (example 153).

Covers the structural-frequency rank, the quadratic-residue prime signature,
the cyclotomy law s_k(p)=gcd(k,p-1)+1, the unitary (Ramanujan) rank, and the
multiplicative conductor-annotated rank A(m).
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from tnfr.errors import TNFRValueError
from tnfr.mathematics.number_theory import (
    _prime_factorization,
    arithmetic_cayley_digraph,
    power_residue_rank,
    power_residue_set,
    quadratic_residue_annotated_rank,
    quadratic_residue_set,
    residue_network_rank,
    unitary_residue_set,
)
from tnfr.physics.structural_diffusion import structural_frequency_rank


def _is_prime(n: int) -> bool:
    return n >= 2 and all(n % d for d in range(2, int(n**0.5) + 1))


# --- structural_frequency_rank (general spectral diagnostic) ---


@pytest.mark.parametrize("n", [4, 5, 6, 7])
def test_structural_frequency_rank_complete_graph_is_two(n):
    # A complete graph has exactly 2 distinct diffusion eigenvalues.
    graph = nx.complete_graph(n, create_using=nx.DiGraph)
    assert structural_frequency_rank(graph) == 2


def test_structural_frequency_rank_matches_residue_network():
    graph = arithmetic_cayley_digraph(7, quadratic_residue_set(7))
    assert structural_frequency_rank(graph) == 3


# --- connection sets ---


def test_quadratic_residue_set_prime():
    assert quadratic_residue_set(7) == {1, 2, 4}


def test_power_residue_set_cubic():
    assert power_residue_set(7, 3) == {pow(x, 3, 7) for x in range(1, 7)} - {0}


def test_unitary_residue_set():
    assert unitary_residue_set(9) == {1, 2, 4, 5, 7, 8}


def test_invalid_modulus_raises():
    with pytest.raises(TNFRValueError):
        quadratic_residue_set(1)


def test_residue_network_rank_unknown_kind_raises():
    with pytest.raises(TNFRValueError):
        residue_network_rank(7, "bogus")


# --- the quadratic-residue prime signature ---


@pytest.mark.parametrize("m", list(range(3, 50, 2)))
def test_qr_rank_three_iff_odd_prime(m):
    assert (residue_network_rank(m, "quadratic") == 3) == _is_prime(m)


@pytest.mark.parametrize("m", list(range(3, 50, 2)))
def test_qr_scalar_rank_equals_annotated_for_odd(m):
    # On the small range the scalar count equals the multiplicative A(m).
    assert residue_network_rank(m, "quadratic") == quadratic_residue_annotated_rank(m)


# --- the cyclotomy law ---


@pytest.mark.parametrize("p", [5, 7, 11, 13, 17, 19, 23, 29, 31])
@pytest.mark.parametrize("k", [2, 3, 4, 5, 6])
def test_cyclotomy_law(p, k):
    assert power_residue_rank(p, k) == math.gcd(k, p - 1) + 1
    assert residue_network_rank(p, "power", k) == power_residue_rank(p, k)


@pytest.mark.parametrize("p", [7, 11, 13, 17, 19, 23])
@pytest.mark.parametrize("k", [3, 4, 5, 6])
def test_complete_splitting_reading(p, k):
    # Maximal rank k+1 is reached iff p splits completely in Q(zeta_k): p=1 mod k.
    assert (residue_network_rank(p, "power", k) == k + 1) == (p % k == 1)


# --- unitary (Ramanujan) rank ---


@pytest.mark.parametrize("p", [5, 7, 11, 13, 17])
def test_unitary_rank_two_for_primes(p):
    assert residue_network_rank(p, "unitary") == 2


# --- annotated multiplicative rank A(m) ---


def test_annotated_rank_prime_power():
    assert quadratic_residue_annotated_rank(9) == 4  # 3^2 -> floor(3*3/2)
    assert quadratic_residue_annotated_rank(27) == 6  # 3^3 -> floor(3*4/2)


@pytest.mark.parametrize("m,omega", [(3, 1), (15, 2), (105, 3), (1155, 4)])
def test_annotated_rank_is_three_to_omega_on_squarefree(m, omega):
    assert quadratic_residue_annotated_rank(m) == 3**omega


def test_annotated_rank_multiplicative_on_coprime():
    a, b = 9, 35  # coprime
    assert quadratic_residue_annotated_rank(a * b) == quadratic_residue_annotated_rank(
        a
    ) * quadratic_residue_annotated_rank(b)


def test_annotated_rank_at_least_tau():
    for m in range(3, 60):
        tau = 1
        for exponent in _prime_factorization(m).values():
            tau *= exponent + 1
        assert quadratic_residue_annotated_rank(m) >= tau


def test_mathematics_package_reexports():
    from tnfr.mathematics import residue_network_rank as reexported

    assert reexported(7, "quadratic") == 3


# --- the cyclotomy law, proved for all k (theory/TNFR_NUMBER_THEORY.md 9.11) ---


def test_cyclotomy_law_large_k():
    """s_k(p) = gcd(k, p-1) + 1 for all k (Gauss-period proof, verified k<=40)."""
    from sympy import isprime

    primes = [p for p in range(3, 64) if isprime(p)]
    for k in range(1, 41):
        for p in primes:
            assert residue_network_rank(p, "power", k) == math.gcd(k, p - 1) + 1


def _conductor_annotated_count(m, decimals=8):
    """#{(F_m(k), gcd(k,m))} with F_m the QR Fourier sum (squares incl 0)."""
    squares = {(x * x) % m for x in range(m)}
    column = np.zeros(m, dtype=complex)
    for residue in squares:
        column[residue] = 1.0
    spectrum = np.fft.fft(column)
    spectrum = np.round(spectrum.real, decimals) + 1j * np.round(
        spectrum.imag, decimals
    )
    return len({(spectrum[t], math.gcd(t, m)) for t in range(m)})


@pytest.mark.parametrize("m", [3, 9, 27, 81, 25, 125])
def test_even_boundary_odd_prime_powers_match(m):
    # Odd prime powers (cyclic unit group): conductor-annotated count == A(p^e).
    assert _conductor_annotated_count(m) == quadratic_residue_annotated_rank(m)


@pytest.mark.parametrize("e", [1, 3, 4, 5])
def test_even_boundary_2adic_diverges(e):
    # 2^e: degenerate at e=1, non-cyclic units for e>=3 -> spectral != A(2^e).
    m = 2**e
    assert _conductor_annotated_count(m) != quadratic_residue_annotated_rank(m)
