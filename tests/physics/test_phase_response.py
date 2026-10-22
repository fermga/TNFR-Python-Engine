"""Independent exact stencils and domain controls for circular phase response."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import pytest

from tnfr.physics.phase_response import derive_phase_response


def _c6_gram():
    first = (Q(1), Q(1, 2), Q(-1, 2), Q(-1), Q(-1, 2), Q(1, 2))
    return tuple(tuple(first[(j - i) % 6] for j in range(6)) for i in range(6))


def _identity(n):
    return tuple(tuple(Q(i == j) for j in range(n)) for i in range(n))


def _stencil(n, coefficients):
    """Materialize a separately derived translation-invariant stencil."""
    return tuple(
        tuple(coefficients.get((j - i) % n, Q(0)) for j in range(n)) for i in range(n)
    )


def _closed_rows(n):
    return tuple(((i - 1) % n, i, (i + 1) % n) for i in range(n))


def _direct_rows(n):
    return tuple((i,) for i in range(n))


def _derive(**overrides):
    arguments = {
        "cosine_gram": _c6_gram(),
        "mean_neighbors": _closed_rows(6),
        "receiver_sources": _direct_rows(6),
        "phase_factor": Q(1, 3),
    }
    arguments.update(overrides)
    return derive_phase_response(**arguments)


def test_c6_closed_mean_has_the_exact_direct_receiver_stencil():
    result = _derive()

    # A closed C6 triple sums to twice its central unit phasor.
    expected_mean = _stencil(6, {0: Q(1, 2), 1: Q(1, 4), 5: Q(1, 4)})
    expected_jacobian = _stencil(6, {0: Q(5, 6), 1: Q(1, 12), 5: Q(1, 12)})
    assert result.mean_resultant_squared == (Q(4),) * 6
    assert result.mean_response == result.receiver_response == expected_mean
    assert result.jacobian == expected_jacobian
    assert result.row_sum_residuals == (Q(0),) * 6
    assert result.is_nonnegative


def test_c6_open_mean_and_open_receivers_reach_distance_two():
    rows = tuple(((i - 1) % 6, (i + 1) % 6) for i in range(6))
    result = _derive(mean_neighbors=rows, receiver_sources=rows)

    # The two neighboring phasors sum to the central unit phasor. Averaging
    # those source responses gives the two-step nearest-neighbor stencil.
    expected_mean = _stencil(6, {1: Q(1, 2), 5: Q(1, 2)})
    expected_received = _stencil(6, {0: Q(1, 2), 2: Q(1, 4), 4: Q(1, 4)})
    expected_jacobian = _stencil(6, {0: Q(5, 6), 2: Q(1, 12), 4: Q(1, 12)})
    assert result.mean_resultant_squared == (Q(1),) * 6
    assert result.mean_response == expected_mean
    assert result.receiver_response == expected_received
    assert result.jacobian == expected_jacobian


def test_c6_closed_receivers_average_all_three_source_proposals():
    result = _derive(receiver_sources=_closed_rows(6))

    expected_received = _stencil(
        6,
        {
            0: Q(1, 3),
            1: Q(1, 4),
            5: Q(1, 4),
            2: Q(1, 12),
            4: Q(1, 12),
        },
    )
    expected_jacobian = _stencil(
        6,
        {
            0: Q(7, 9),
            1: Q(1, 12),
            5: Q(1, 12),
            2: Q(1, 36),
            4: Q(1, 36),
        },
    )
    assert result.receiver_response == expected_received
    assert result.jacobian == expected_jacobian
    assert result.jacobian[0][3] == 0


def test_nonuniform_receiver_counts_have_an_independently_solved_matrix():
    result = _derive(
        cosine_gram=((1, 1, 1),) * 3,
        mean_neighbors=((0, 1), (1, 2), (2,)),
        receiver_sources=((0, 1), (1, 2), (0, 2)),
        phase_factor=Q(1, 2),
    )
    assert result.mean_resultant_squared == (4, 4, 1)
    assert result.mean_response == (
        (Q(1, 2), Q(1, 2), 0),
        (0, Q(1, 2), Q(1, 2)),
        (0, 0, 1),
    )
    assert result.receiver_response == (
        (Q(1, 4), Q(1, 2), Q(1, 4)),
        (0, Q(1, 4), Q(3, 4)),
        (Q(1, 4), Q(1, 4), Q(1, 2)),
    )
    assert result.jacobian == (
        (Q(5, 8), Q(1, 4), Q(1, 8)),
        (0, Q(5, 8), Q(3, 8)),
        (Q(1, 8), Q(1, 8), Q(3, 4)),
    )


def test_rational_planar_phasors_give_an_exact_nonsymmetric_mean_row():
    # Unit vectors (1,0), (3/5,4/5), (-4/5,3/5) have sum (4/5,7/5).
    gram = ((1, Q(3, 5), Q(-4, 5)), (Q(3, 5), 1, 0), (Q(-4, 5), 0, 1))
    result = _derive(
        cosine_gram=gram,
        mean_neighbors=((0, 1, 2),) * 3,
        receiver_sources=_direct_rows(3),
        phase_factor=1,
    )
    expected = ((Q(4, 13), Q(8, 13), Q(1, 13)),) * 3
    assert result.mean_resultant_squared == (Q(13, 5),) * 3
    assert (
        result.mean_response == result.receiver_response == result.jacobian == expected
    )


@pytest.mark.parametrize("factor", (Q(1, 2), Q(1)))
def test_antipodal_bridge_has_negative_response_despite_rotation_covariance(factor):
    signs = (1, 1, 1, -1, -1, -1)
    gram = tuple(tuple(a * b for b in signs) for a in signs)
    rows = ((0, 1, 3),) + _direct_rows(6)[1:]
    result = _derive(cosine_gram=gram, mean_neighbors=rows, phase_factor=factor)

    # The bridge mean is Arg(1+1-1); its three nonzero derivatives are 1,1,-1.
    assert result.mean_resultant_squared == (1,) * 6
    assert result.mean_response[0] == (1, 1, 0, -1, 0, 0)
    assert result.jacobian[0] == (1, factor, 0, -factor, 0, 0)
    assert result.jacobian[1:] == _identity(6)[1:]
    assert result.row_sum_residuals == (0,) * 6
    assert not result.is_nonnegative


def test_zero_factor_classifies_the_final_jacobian_not_its_negative_mean():
    signs = (1, 1, 1, -1, -1, -1)
    result = _derive(
        cosine_gram=tuple(tuple(a * b for b in signs) for a in signs),
        mean_neighbors=((0, 1, 3),) + _direct_rows(6)[1:],
        phase_factor=0,
    )
    assert result.mean_response[0][3] == -1
    assert result.jacobian == _identity(6)
    assert result.is_nonnegative


def test_a_small_positive_resultant_is_retained_without_a_tolerance_cutoff():
    gram = ((1, Q(-999, 1001)), (Q(-999, 1001), 1))
    result = _derive(
        cosine_gram=gram,
        mean_neighbors=((0, 1),) * 2,
        receiver_sources=_direct_rows(2),
        phase_factor=1,
    )
    assert result.mean_resultant_squared == (Q(4, 1001),) * 2
    assert result.jacobian == ((Q(1, 2), Q(1, 2)),) * 2


@pytest.mark.parametrize("factor", (Q(0), Q(1)))
def test_a_zero_phasor_resultant_is_undefined_even_at_zero_factor(factor):
    with pytest.raises(ValueError, match="nonzero resultant"):
        _derive(
            cosine_gram=((1, -1), (-1, 1)),
            mean_neighbors=((0, 1),) * 2,
            receiver_sources=_direct_rows(2),
            phase_factor=factor,
        )


@pytest.mark.parametrize(
    "gram",
    (
        (),
        ((1, 0),),
        ((1, 0), (0,)),
        ((1, 0), (Q(1, 2), 1)),
        ((Q(4, 5), 0), (0, 1)),
        ((1, 2), (2, 1)),
        ((1, 0, 0), (0, 1, 2), (0, 2, 1)),
        ((1, 1, 1), (1, 1, -1), (1, -1, 1)),
        _identity(3),
    ),
)
def test_nonunit_nonplanar_or_invalid_gram_data_are_not_repaired(gram):
    with pytest.raises(ValueError):
        _derive(
            cosine_gram=gram,
            mean_neighbors=_direct_rows(len(gram)),
            receiver_sources=_direct_rows(len(gram)),
        )


@pytest.mark.parametrize(
    "value", (True, "1", complex(1, 0), float("nan"), float("inf"))
)
def test_invalid_gram_scalars_are_rejected(value):
    with pytest.raises((TypeError, ValueError)):
        _derive(
            cosine_gram=((value,),), mean_neighbors=((0,),), receiver_sources=((0,),)
        )


@pytest.mark.parametrize(
    "factor", (-1, Q(1001, 1000), True, False, "0.5", float("nan"), float("inf"))
)
def test_invalid_phase_factors_are_rejected(factor):
    with pytest.raises((TypeError, ValueError)):
        _derive(phase_factor=factor)


def test_represented_half_and_exact_half_have_identical_coefficients():
    assert _derive(phase_factor=0.5) == _derive(phase_factor=Q(1, 2))


def test_a_nonnegative_jacobian_can_preserve_every_phase_disagreement():
    result = _derive(mean_neighbors=_direct_rows(6), phase_factor=1)
    assert result.jacobian == _identity(6)
    assert result.is_nonnegative
    assert result.row_sum_residuals == (0,) * 6


@pytest.mark.parametrize("field", ("mean_neighbors", "receiver_sources"))
@pytest.mark.parametrize(
    "rows",
    (
        (),
        ((0,),) * 5,
        ((),) + _direct_rows(6)[1:],
        ((0, 0),) + _direct_rows(6)[1:],
        ((-1,),) + _direct_rows(6)[1:],
        ((6,),) + _direct_rows(6)[1:],
        ((True,),) + _direct_rows(6)[1:],
        ((Q(0),),) + _direct_rows(6)[1:],
        ((0.0,),) + _direct_rows(6)[1:],
    ),
)
def test_row_dimensions_membership_and_index_types_are_explicit(field, rows):
    with pytest.raises(ValueError):
        _derive(**{field: rows})


@pytest.mark.parametrize("field", ("mean_neighbors", "receiver_sources", "cosine_gram"))
@pytest.mark.parametrize("unordered", ("012", {0: (0,)}, {(0,)}))
def test_unordered_or_textual_outer_inputs_are_rejected(field, unordered):
    with pytest.raises(TypeError):
        _derive(**{field: unordered})


def test_node_permutation_covariance_preserves_the_declared_source_incidence():
    rows = ((0, 1), (1, 2, 3), (2,), (3, 4), (4, 5, 0), (5,))
    receivers = ((0, 1), (1,), (1, 2), (2, 3), (3, 4, 5), (0, 5))
    original = _derive(
        mean_neighbors=rows, receiver_sources=receivers, phase_factor=Q(2, 7)
    )
    permutation = (3, 0, 5, 2, 1, 4)
    inverse = {old: new for new, old in enumerate(permutation)}
    gram = _c6_gram()
    permuted = _derive(
        cosine_gram=tuple(tuple(gram[i][j] for j in permutation) for i in permutation),
        mean_neighbors=tuple(tuple(inverse[j] for j in rows[i]) for i in permutation),
        receiver_sources=tuple(
            tuple(inverse[j] for j in receivers[i]) for i in permutation
        ),
        phase_factor=Q(2, 7),
    )
    for field in ("mean_response", "receiver_response", "jacobian"):
        matrix = getattr(original, field)
        assert getattr(permuted, field) == tuple(
            tuple(matrix[i][j] for j in permutation) for i in permutation
        )
    assert permuted.mean_resultant_squared == tuple(
        original.mean_resultant_squared[i] for i in permutation
    )
    assert permuted.is_nonnegative == original.is_nonnegative
    assert permuted.row_sum_residuals == original.row_sum_residuals == (0,) * 6


def test_materialization_is_readonly_and_returned_reference_is_deeply_immutable():
    gram = [list(row) for row in _c6_gram()]
    means = [list(row) for row in _closed_rows(6)]
    receivers = [list(row) for row in _direct_rows(6)]
    inputs_before = deepcopy((gram, means, receivers))
    result = _derive(cosine_gram=gram, mean_neighbors=means, receiver_sources=receivers)
    assert (gram, means, receivers) == inputs_before
    gram[0][0] = 99
    means[0].append(4)
    receivers[0].append(1)
    assert result.cosine_gram == _c6_gram()
    assert result.mean_neighbors == _closed_rows(6)
    assert result.receiver_sources == _direct_rows(6)
    with pytest.raises(FrozenInstanceError):
        result.phase_factor = Q(0)
    with pytest.raises(TypeError):
        result.jacobian[0][0] = Q(0)
    assert all(isinstance(value, Q) for row in result.jacobian for value in row)
