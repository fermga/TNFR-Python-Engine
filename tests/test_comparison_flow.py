"""Exact independent controls for the componentwise comparison exponential."""

from fractions import Fraction as Q
from math import factorial

import pytest

from tnfr.mathematics._comparison_flow import comparison_flow_upper


def _diagonal(values):
    return tuple(
        tuple(value if i == j else Q(0) for j in range(4))
        for i, value in enumerate(values)
    )


def _negative_exp_reference(value):
    """Independent alternating degree 96/97 bounds on exp(-value), 0<=value<=1."""
    assert 0 <= value <= 1
    upper = sum(((-value) ** k / factorial(k) for k in range(97)), Q(0))
    lower = upper - value**97 / factorial(97)
    return lower, upper


def test_zero_duration_keeps_exact_input_radii_with_outward_materialization():
    radii = (Q(1, 3), Q(0), Q(2, 7), Q(1, 1 << 200))
    result = comparison_flow_upper(_diagonal((-100, 0, 1, 8)), radii, Q(0))
    assert all(
        value <= bound <= value + Q(1, 1 << 128) for value, bound in zip(radii, result)
    )
    assert result[1] == 0 and result[3] > 0


def test_zero_uncertainty_is_preserved_exactly():
    matrix = tuple(
        tuple(Q(-1, 2) if i == j else Q(1, 8) for j in range(4)) for i in range(4)
    )
    assert comparison_flow_upper(matrix, (0, 0, 0, 0), Q(1)) == (0, 0, 0, 0)


def test_negative_diagonal_retains_contraction_instead_of_absolute_growth():
    radii = (Q(1), Q(2), Q(3), Q(4))
    result = comparison_flow_upper(_diagonal((-1,) * 4), radii, Q(1))
    _, exp_upper = _negative_exp_reference(Q(1))
    for radius, bound in zip(radii, result):
        assert radius * exp_upper <= bound < radius
        assert bound - radius * exp_upper < Q(1, 10**35)


def test_shifted_nilpotent_chain_matches_finite_polynomial_times_decay():
    matrix = tuple(
        tuple(Q(-1) if i == j else Q(int(j == i + 1)) for j in range(4))
        for i in range(4)
    )
    radii, h = (Q(1), Q(2), Q(0), Q(4)), Q(2, 3)
    result = comparison_flow_upper(matrix, radii, h)
    _, decay_upper = _negative_exp_reference(h)
    for i, bound in enumerate(result):
        exact_polynomial = sum(
            (h**k * radii[i + k] / factorial(k) for k in range(4 - i)), Q(0)
        )
        reference = exact_polynomial * decay_upper
        assert reference <= bound < reference + Q(1, 10**34)


def test_positive_diagonal_at_unit_work_limit_encloses_independent_exp_series():
    result = comparison_flow_upper(_diagonal((1,) * 4), (1, 2, 3, 4), Q(1))
    reference = sum((Q(1, factorial(k)) for k in range(101)), Q(0))
    reference_upper = reference + Q(1, factorial(101)) / (1 - Q(1, 102))
    for radius, bound in zip((1, 2, 3, 4), result):
        assert radius * reference_upper <= bound
        assert bound - radius * reference_upper < Q(1, 10**34)


def test_dense_metzler_matrix_matches_independent_projector_decomposition():
    # P=ones/4 is an exact projector. exp(h*(P-2Id)) is therefore
    # exp(-2h)*Id + [exp(-h)-exp(-2h)]*P, including nonuniform input.
    matrix = tuple(
        tuple(Q(1, 4) - (2 if i == j else 0) for j in range(4)) for i in range(4)
    )
    radii = (Q(0), Q(1), Q(2), Q(3))
    result = comparison_flow_upper(matrix, radii, Q(1, 2))
    low_one, high_one = _negative_exp_reference(Q(1))
    _, high_half = _negative_exp_reference(Q(1, 2))
    mean = sum(radii) / 4
    for radius, bound in zip(radii, result):
        coefficient = radius - mean
        reference_upper = mean * high_half + coefficient * (
            high_one if coefficient >= 0 else low_one
        )
        assert reference_upper <= bound < reference_upper + Q(1, 10**34)


def test_comparison_encloses_exact_nonlinear_flow_separation_on_convex_tube():
    # xdot=-x*x has exact solution x/(1+h*x). At h=1/4 both solutions
    # from x in [1,2] and center 3/2 remain in [1/2,2], where J=-2*x<=-1.
    # This supplies an independent dynamical use of the signed comparison.
    h, center = Q(1, 4), Q(3, 2)
    bound = comparison_flow_upper(_diagonal((-1,) * 4), (Q(1, 2),) * 4, h)[0]
    center_endpoint = center / (1 + h * center)
    for initial in (Q(1), Q(2)):
        assert abs(initial / (1 + h * initial) - center_endpoint) <= bound
    assert bound < Q(1, 2)


@pytest.mark.parametrize(
    "matrix,radii,duration,message",
    (
        (((0,) * 4,) * 3, (0,) * 4, Q(1), "four by four"),
        (((0,) * 3,) * 4, (0,) * 4, Q(1), "four by four"),
        (_diagonal((0,) * 4), (0,) * 3, Q(1), "four values"),
        (
            ((0, -1, 0, 0), (0,) * 4, (0,) * 4, (0,) * 4),
            (1,) * 4,
            Q(1),
            "off-diagonals",
        ),
        (_diagonal((0,) * 4), (-1, 0, 0, 0), Q(1), "nonnegative"),
        (_diagonal((0,) * 4), (1,) * 4, Q(-1), "nonnegative"),
        (_diagonal((2,) * 4), (1,) * 4, Q(1), "norm"),
        (_diagonal((-2,) * 4), (1,) * 4, Q(1), "alpha"),
    ),
)
def test_invalid_comparison_or_unresolved_numerical_domain_rejects(
    matrix, radii, duration, message
):
    with pytest.raises(ValueError, match=message):
        comparison_flow_upper(matrix, radii, duration)


@pytest.mark.parametrize("invalid", (False, 0.0, "0"))
def test_raw_nonrational_coordinates_are_not_silently_materialized(invalid):
    with pytest.raises(TypeError, match="exact Fraction"):
        comparison_flow_upper(_diagonal((0,) * 4), (0, 0, 0, invalid), Q(1))
    with pytest.raises(TypeError, match="exact Fraction"):
        comparison_flow_upper(_diagonal((invalid, 0, 0, 0)), (0,) * 4, Q(1))
    with pytest.raises(TypeError, match="exact Fraction"):
        comparison_flow_upper(_diagonal((0,) * 4), (0,) * 4, invalid)
