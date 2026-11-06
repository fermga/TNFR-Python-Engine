r"""Tests for the R2 arithmetic pulse recurrence.

On the pointed k-th power residue Cayley network ``(G_{p,k}, 0)`` the pulse
moments ``μ_m = e₀ᵀ L^m e₀`` have Hankel rank = Krylov dimension = #distinct
eigenvalues = ``gcd(k, p−1) + 1`` for primes ``p`` — an exact-rational
identity.
Composites are controls outside the theorem.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from math import gcd

import numpy as np
import pytest

from tnfr.mathematics.arithmetic_pulse import (
    cyclotomic_rank,
    pointed_pulse_hankel_rank,
    pointed_pulse_krylov_dimension,
    power_residue_laplacian,
    pulse_recurrence_matches_cyclotomy,
)
from tnfr.mathematics.cayley import (
    cayley_action,
    cayley_diffusion_action,
    cayley_first_row,
    cayley_laplacian,
    cayley_spectrum,
)
from tnfr.mathematics.krylov import (
    exact_rank,
    hankel_rank,
    krylov_dimension,
    moment_sequence,
)
from tnfr.mathematics.number_theory import arithmetic_cayley_digraph, power_residue_set
from tnfr.physics.structural_diffusion import structural_diffusion_operator

# Distinct ranks/supports: full units, directed and reciprocal quadratics,
# cubic/quartic/sextic images, and a singleton connection (full cycle rank).
PRIME_CASES = [(5, 1), (5, 2), (7, 2), (7, 3), (11, 3), (13, 4), (19, 6), (7, 6)]


@pytest.fixture(scope="module")
def pulse_ranks():
    """Compute the two independent exact producers once per requested network."""

    @lru_cache(maxsize=None)
    def observe(modulus, power):
        return (
            pointed_pulse_hankel_rank(modulus, power),
            pointed_pulse_krylov_dimension(modulus, power),
        )

    return observe


def test_shared_cayley_builder_matches_arithmetic_pulse():
    connection = set(power_residue_set(11, 3))
    assert cayley_laplacian(11, connection) == power_residue_laplacian(11, 3)


def test_exact_cayley_action_and_spectrum_match_dense_reference():
    connection = {1, 2}
    vector = [Fraction(i - 2) for i in range(5)]
    matrix = cayley_laplacian(5, connection)
    expected = [
        sum((row[j] * vector[j] for j in range(5)), Fraction(0)) for row in matrix
    ]
    assert cayley_first_row(5, connection) == matrix[0]
    assert cayley_action(5, connection, vector) == expected
    dense_eigenvalues = np.linalg.eigvals(
        np.array([[float(value) for value in row] for row in matrix])
    )
    spectrum = cayley_spectrum(5, connection)
    assert all(
        min(abs(value - candidate) for candidate in dense_eigenvalues) < 1e-10
        for value in spectrum
    )


@pytest.mark.parametrize("modulus", [7, 8, 9, 15])
def test_cayley_action_covers_prime_even_prime_power_and_composite(modulus):
    connection = {1, 2}
    vector = [Fraction(index) for index in range(modulus)]
    matrix = cayley_laplacian(modulus, connection)
    expected = [
        sum(
            (row[index] * vector[index] for index in range(modulus)),
            Fraction(0),
        )
        for row in matrix
    ]
    assert cayley_action(modulus, connection, vector) == expected


def test_cayley_action_does_not_materialize_dense_matrix(monkeypatch):
    import tnfr.mathematics.cayley as module

    monkeypatch.setattr(
        module,
        "cayley_laplacian",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError()),
    )
    assert module.cayley_action(5, {1, 2}, [Fraction(i) for i in range(5)])


def test_cayley_representations_reject_empty_connection():
    with pytest.raises(ValueError):
        cayley_action(5, set(), [Fraction(i) for i in range(5)])
    with pytest.raises(ValueError):
        cayley_first_row(5, {0, 5})


def test_cayley_diffusion_action_matches_dense_nodal_transport():
    modulus = 7
    connection = {1, 2}
    vector = np.array([1.0, -1.0, 0.5, 0.0, 2.0, -0.25, 0.75])
    structural_time = 0.6
    capacity = 0.7
    matrix = np.array(
        [
            [float(value) for value in row]
            for row in cayley_laplacian(modulus, connection)
        ]
    )
    from tnfr.physics.spectral_projectors import matrix_exponential

    expected = matrix_exponential(-capacity * structural_time * matrix) @ vector
    actual = cayley_diffusion_action(
        modulus,
        connection,
        vector,
        structural_time=structural_time,
        capacity=capacity,
    )
    assert actual == pytest.approx(expected, abs=1e-10)


def test_cayley_diffusion_action_handles_frozen_and_invalid_capacity():
    vector = np.array([1.0, -0.5, 2.0, 0.25, -1.0])
    assert cayley_diffusion_action(5, {1, 2}, vector, capacity=0.0) == pytest.approx(
        vector
    )
    with pytest.raises(ValueError, match="capacity"):
        cayley_diffusion_action(5, {1, 2}, vector, capacity=-1.0)
    with pytest.raises(ValueError, match="structural_time"):
        cayley_diffusion_action(5, {1, 2}, vector, structural_time=-1.0)


# --- exact Krylov / Hankel primitives -----------------------------------------


def test_exact_rank_of_known_matrix():
    F = Fraction
    # rank 2: third row is row1 + row2
    M = [[F(1), F(0)], [F(0), F(1)], [F(1), F(1)]]
    assert exact_rank(M) == 2
    assert exact_rank([[F(0), F(0)], [F(0), F(0)]]) == 0


def test_moment_sequence_is_exact_rational():
    F = Fraction
    L = [[F(1), F(-1, 2)], [F(-1, 2), F(1)]]
    v = [F(1), F(0)]
    moments = moment_sequence(L, v, 3)
    assert all(isinstance(m, Fraction) for m in moments)
    assert moments[0] == F(1)  # v^T v
    assert moments[1] == F(1)  # v^T L v = L[0][0]


def test_hankel_equals_krylov_on_a_small_operator():
    F = Fraction
    L = [[F(2), F(0), F(0)], [F(0), F(2), F(0)], [F(0), F(0), F(5)]]
    v = [F(1), F(1), F(1)]  # excites eigenvalues {2, 5} -> rank 2
    assert krylov_dimension(L, v) == 2
    assert hankel_rank(L, v) == 2


# --- the pulse recurrence theorem (primes) ------------------------------------


@pytest.mark.parametrize("p,k", PRIME_CASES)
def test_pulse_rank_equals_cyclotomy_for_primes(p, k, pulse_ranks):
    hankel, _ = pulse_ranks(p, k)
    assert hankel == gcd(k, p - 1) + 1


@pytest.mark.parametrize("p,k", PRIME_CASES)
def test_hankel_equals_krylov(p, k, pulse_ranks):
    hankel, krylov = pulse_ranks(p, k)
    assert hankel == krylov


@pytest.mark.parametrize("modulus,power,expected", [(7, 3, True), (9, 2, False)])
def test_pulse_comparison_wrapper_retains_prime_and_composite_scope(
    modulus, power, expected
):
    assert pulse_recurrence_matches_cyclotomy(modulus, power) is expected


@pytest.mark.parametrize("p,k", [(7, 2), (11, 3), (13, 4), (19, 6)])
def test_spectral_agreement(p, k, pulse_ranks):
    """Krylov dimension == number of distinct eigenvalues (e0 excites all
    Fourier modes of the circulant)."""
    _, L = structural_diffusion_operator(
        arithmetic_cayley_digraph(p, power_residue_set(p, k))
    )
    distinct = len(np.unique(np.round(np.linalg.eigvals(L), 6)))
    _, krylov = pulse_ranks(p, k)
    assert krylov == distinct


@pytest.mark.parametrize("p,k", [(11, 2), (13, 3), (17, 4)])
def test_point_relabel_equivalence(p, k):
    """The pointed-pulse rank is independent of the chosen point (the network is
    a Cayley graph, so translation is an automorphism)."""
    L = power_residue_laplacian(p, k)
    dims = set()
    for t in (0, 3, 7):
        e = [Fraction(0)] * p
        e[t % p] = Fraction(1)
        dims.add(krylov_dimension(L, e))
    assert len(dims) == 1


# --- composite controls (outside the theorem) ---------------------------------


@pytest.mark.parametrize("n,k", [(9, 2), (15, 2), (25, 3)])
def test_composite_controls_are_kronecker_but_not_cyclotomic(n, k, pulse_ranks):
    # Kronecker (Hankel == Krylov) still holds for composites ...
    hankel, krylov = pulse_ranks(n, k)
    assert hankel == krylov
    # ... but the cyclotomy value gcd(k, n-1)+1 does NOT: the identity is
    # prime-specific (this is a control, not a primality test).
    assert hankel != cyclotomic_rank(n, k)
