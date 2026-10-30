"""Independent exact current, symmetry and rank controls for static boundaries."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as Q
from itertools import repeat
from math import comb

import pytest

from tnfr.mathematics.krylov import exact_rank
from tnfr.physics._phase_boundary_information import (
    _admit_fourier_current,
    _derive_boundary_information,
    _evaluate_boundary_response,
)


def _unit(t):
    return ((1 - t * t) / (1 + t * t), 2 * t / (1 + t * t))


def _expanded_power(pair, n):
    """Binomial expansion independent of the production squaring recurrence."""
    c, s = pair
    real = sum(
        (
            Q((-1) ** (k // 2) * comb(n, k)) * c ** (n - k) * s**k
            for k in range(0, n + 1, 2)
        ),
        Q(0),
    )
    imaginary = sum(
        (
            Q((-1) ** ((k - 1) // 2) * comb(n, k)) * c ** (n - k) * s**k
            for k in range(1, n + 1, 2)
        ),
        Q(0),
    )
    return real, imaginary


def _difference(gap, query):
    return (
        gap[0] * query[0] + gap[1] * query[1],
        gap[1] * query[0] - gap[0] * query[1],
    )


def _rotate(pair, rotation):
    return (
        pair[0] * rotation[0] - pair[1] * rotation[1],
        pair[0] * rotation[1] + pair[1] * rotation[0],
    )


def _direct_current(coefficients, gaps, query):
    return sum(
        (
            scale * _expanded_power(_difference(gap, query), n)[1]
            for gap in gaps
            for n, scale in coefficients
        ),
        Q(0),
    )


def _evaluate(coefficients, gaps, query=(1, 0)):
    return _evaluate_boundary_response(
        coefficients, gap_phasors=gaps, query_phasor=query
    )


def test_sparse_law_metadata_keeps_exact_sign_and_raw_order():
    coefficients = ((7, Q(-5, 3)), (2, Q(0)), (1, Q(3, 5)))
    info = _derive_boundary_information(coefficients)
    assert info.coefficients == ((7, Q(-5, 3)), (1, Q(3, 5)))
    assert info.minimum_regular_dimension == 4
    assert info.witness_degree == 15
    assert info.signed_frequency_eigenvalues == (
        (7, Q(-175, 2)),
        (-7, Q(-175, 2)),
        (1, Q(9, 2)),
        (-1, Q(9, 2)),
    )
    with pytest.raises(FrozenInstanceError):
        info.minimum_regular_dimension = 1


def test_exact_nonzero_support_survives_binary64_underflow_and_overflow():
    tiny, huge = Q(1, 10**1000), Q(10**500)
    info = _derive_boundary_information(((1, tiny), (128, -huge)))
    assert info.minimum_regular_dimension == 4
    assert info.witness_degree == 257
    assert info.signed_frequency_eigenvalues == (
        (1, 257 * tiny / 2),
        (-1, 257 * tiny / 2),
        (128, -257 * 64 * huge),
        (-128, -257 * 64 * huge),
    )
    assert _evaluate(((1, tiny),), ((0, 1),)).current == tiny


@pytest.mark.parametrize("coefficients", [(), ((128, 0), (1, -0.0))])
def test_zero_law_has_zero_dimension_without_a_grid_witness(coefficients):
    info = _derive_boundary_information(coefficients)
    assert info.coefficients == ()
    assert info.minimum_regular_dimension == info.witness_degree == 0
    assert info.signed_frequency_eigenvalues == ()
    response = _evaluate(coefficients, ((0, 1), (1, 0)))
    assert response.degree == 2
    assert response.moments == response.harmonic_contributions == ()
    assert response.current == 0


def test_empty_boundary_preserves_active_law_and_zero_moments():
    response = _evaluate(((3, Q(-7, 8)), (1, 5)), ())
    assert response.degree == 0
    assert response.moments == ((3, (Q(0), Q(0))), (1, (Q(0), Q(0))))
    assert response.current == 0


def test_moment_response_agrees_with_independent_gapwise_expansion():
    coefficients = ((5, Q(-7, 13)), (2, Q(2, 9)), (1, Q(3, 11)))
    gaps = tuple(_unit(Q(t, 5)) for t in (-4, -1, 0, 2, 9))
    query = _unit(Q(4, 7))
    response = _evaluate(coefficients, gaps, query)
    assert response.degree == len(gaps)
    assert response.current == _direct_current(coefficients, gaps, query)
    assert response.current != 0
    for harmonic, moment in response.moments:
        expanded = tuple(_expanded_power(gap, harmonic) for gap in gaps)
        assert moment == (
            sum((pair[0] for pair in expanded), Q(0)),
            sum((pair[1] for pair in expanded), Q(0)),
        )


def test_gap_minus_query_sign_and_unnormalized_multiplicity():
    assert _evaluate(((1, 7),), ((0, 1),), (1, 0)).current == 7
    assert _evaluate(((1, 7),), ((1, 0),), (0, 1)).current == -7
    doubled = _evaluate(((1, 7),), ((0, 1), (0, 1)), (1, 0))
    assert doubled.degree == 2
    assert doubled.current == 14


def test_permutation_additivity_and_common_phase_covariance():
    coefficients = ((2, Q(4, 7)), (5, Q(-8, 3)))
    gaps = tuple(_unit(Q(t, 3)) for t in (-4, -1, 0, 1, 2))
    query, rotation = _unit(Q(1, 9)), _unit(Q(5, 8))
    response = _evaluate(coefficients, gaps, query)
    reversed_response = _evaluate(coefficients, reversed(gaps), query)
    assert reversed_response.moments == response.moments
    assert reversed_response.current == response.current
    left = _evaluate(coefficients, gaps[:2], query)
    right = _evaluate(coefficients, gaps[2:], query)
    assert response.current == left.current + right.current
    assert response.degree == left.degree + right.degree
    for (n, total), (nl, ml), (nr, mr) in zip(
        response.moments, left.moments, right.moments
    ):
        assert n == nl == nr
        assert total == (ml[0] + mr[0], ml[1] + mr[1])
    rotated = _evaluate(
        coefficients,
        tuple(_rotate(gap, rotation) for gap in gaps),
        _rotate(query, rotation),
    )
    assert rotated.current == response.current
    for (n, before), (nr, after) in zip(response.moments, rotated.moments):
        assert n == nr
        assert after == _rotate(before, _expanded_power(rotation, n))


def test_reflection_reverses_current_and_zero_moments_remain_available():
    coefficients = ((1, 3), (2, Q(-2, 7)), (3, 5))
    gaps = (_unit(Q(2, 3)), _unit(Q(-1, 4)))
    query = _unit(Q(3, 4))
    reflected = _evaluate(
        coefficients,
        tuple((c, -s) for c, s in gaps),
        (query[0], -query[1]),
    )
    assert reflected.current == -_evaluate(coefficients, gaps, query).current
    cancelled = _evaluate(((1, 1), (3, 2)), ((1, 0), (-1, 0)))
    assert cancelled.degree == 2
    assert cancelled.moments == ((1, (0, 0)), (3, (0, 0)))
    assert cancelled.current == 0
    assert (
        _derive_boundary_information(cancelled.coefficients).minimum_regular_dimension
        == 4
    )


@pytest.mark.parametrize(
    "coefficients, rank", [(((2, Q(-2, 5)),), 2), (((2, 3), (5, Q(-4, 9)), (7, 1)), 6)]
)
def test_independent_rational_current_jacobian_has_full_claimed_rank(
    coefficients, rank
):
    # A rational configuration provides a separate rank witness, without
    # approximating the irrational regular grid used by the theorem.
    phasors = tuple(_unit(Q(t, 5)) for t in range(-3, 4))
    jacobian = tuple(
        tuple(
            sum(
                (
                    n * scale * _expanded_power(_difference(gap, query), n)[0]
                    for n, scale in coefficients
                ),
                Q(0),
            )
            for gap in phasors
        )
        for query in phasors
    )
    assert exact_rank(jacobian) == rank
    assert _derive_boundary_information(coefficients).minimum_regular_dimension == rank


def test_degenerate_boundary_does_not_assert_pointwise_full_rank():
    coefficients = ((2, Q(3, 7)), (5, Q(-4, 9)))
    derivative = sum((n * scale for n, scale in coefficients), Q(0))
    assert exact_rank(((derivative,) * 5,) * 5) == 1
    assert _derive_boundary_information(coefficients).minimum_regular_dimension == 4


@pytest.mark.parametrize("scale", [Q(7, 3), Q(-11, 8)])
def test_three_point_cosine_jacobian_independently_has_reported_spectrum(scale):
    # At gaps and queries 0, 2*pi/3, 4*pi/3, the sine derivative is exactly
    # b on the diagonal and -b/2 elsewhere, despite irrational sine values.
    jacobian = tuple(
        tuple(scale if i == j else -scale / 2 for j in range(3)) for i in range(3)
    )
    info = _derive_boundary_information(((1, scale),))
    eigenvalue = info.signed_frequency_eigenvalues[0][1]
    for vector in ((2, -1, -1), (0, 1, -1)):
        assert tuple(
            sum(a * b for a, b in zip(row, vector)) for row in jacobian
        ) == tuple(eigenvalue * value for value in vector)
    assert all(sum(row) == 0 for row in jacobian)
    assert exact_rank(jacobian) == info.minimum_regular_dimension == 2


@pytest.mark.parametrize("harmonic", [1, 2, 7, 128])
def test_unit_consensus_slope_does_not_select_a_unique_two_coordinate_law(harmonic):
    info = _derive_boundary_information(((harmonic, Q(1, harmonic)),))
    # Direct differentiation at zero gives j'(0)=n*(1/n)=1 for each law.
    assert sum(n * b for n, b in info.coefficients) == 1
    assert info.minimum_regular_dimension == 2
    assert info.witness_degree == 2 * harmonic + 1


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 129, 1.0, Q(1), "1", 1j])
def test_harmonic_domain_is_integral_positive_bounded_and_nonboolean(invalid):
    with pytest.raises((TypeError, ValueError)):
        _derive_boundary_information(((invalid, 1),))


@pytest.mark.parametrize(
    "invalid", [True, False, float("nan"), float("inf"), -float("inf"), "1", 1j]
)
def test_coefficient_domain_uses_shared_real_admission(invalid):
    with pytest.raises((TypeError, ValueError)):
        _derive_boundary_information(((1, invalid),))


@pytest.mark.parametrize(
    "invalid", [None, "", {}, {(1, 2)}, ((1,),), ((1, 2, 3),), ({1: 2},)]
)
def test_coefficient_container_must_have_ordered_pairs(invalid):
    with pytest.raises((TypeError, ValueError)):
        _derive_boundary_information(invalid)


@pytest.mark.parametrize("coefficients", [((1, 1), (1, 2)), ((1, 0), (1, 0))])
def test_duplicate_harmonics_reject_even_zero_rows(coefficients):
    with pytest.raises(ValueError, match="distinct"):
        _admit_fourier_current(coefficients)


def test_iterable_budgets_are_bounded_before_materialization():
    with pytest.raises(ValueError, match="128-row"):
        _derive_boundary_information(repeat((1, 0)))
    with pytest.raises(ValueError, match="1..128"):
        _evaluate((), repeat((1, 0)))
    with pytest.raises(ValueError, match="harmonic and scale"):
        _derive_boundary_information((repeat(1),))
    with pytest.raises(ValueError, match="cosine/sine"):
        _evaluate((), (), repeat(1))


@pytest.mark.parametrize(
    "bad",
    [(0, 0), (True, 0), (0.6, 0.8), (Q(3, 5), Q(3, 5)), (float("nan"), 0), (1, 0, 0)],
)
def test_zero_law_still_admits_gap_and_query_primitives(bad):
    with pytest.raises((TypeError, ValueError)):
        _evaluate((), (bad,))
    with pytest.raises((TypeError, ValueError)):
        _evaluate((), (), bad)


def test_zero_law_rejects_unordered_or_cached_input():
    with pytest.raises(TypeError):
        _evaluate((), {(1, 0)})
    with pytest.raises(TypeError):
        _evaluate((), (), {1, 0})
    report = _derive_boundary_information(((1, 1),))
    with pytest.raises(TypeError):
        _derive_boundary_information(report)
    with pytest.raises(TypeError):
        _evaluate(report, ((1, 0),))


def test_represented_floats_and_generator_inputs_preserve_admitted_values():
    response = _evaluate(
        ((n, b) for n, b in ((1, 0.1), (2, -0.0))), ((1.0, 0.0),), (0.0, 1.0)
    )
    assert response.coefficients == ((1, Q.from_float(0.1)),)
    assert response.current == -Q.from_float(0.1)


def test_maximum_computational_boundaries_remain_exact():
    coefficients = tuple((n, Q(1, n)) for n in range(1, 129))
    response = _evaluate(coefficients, ((0, 1),) * 128)
    expected = 128 * sum((Q((-1) ** ((n - 1) // 2), n) for n in range(1, 129, 2)), Q(0))
    assert response.degree == 128
    assert len(response.moments) == 128
    assert response.current == expected
    assert _derive_boundary_information(coefficients).minimum_regular_dimension == 256
