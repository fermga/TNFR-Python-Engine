"""Exact geometric controls for the local circular phase-source level set."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q

import pytest

from tnfr.physics.phase_response import (
    derive_phase_response,
    observe_phase_source_geometry,
)


def _derive(vectors, neighbors, *, factor=Q(1), receivers=None):
    gram = tuple(
        tuple(sum((a * b for a, b in zip(left, right)), Q(0)) for right in vectors)
        for left in vectors
    )
    return derive_phase_response(
        cosine_gram=gram,
        mean_neighbors=neighbors,
        receiver_sources=receivers or tuple((i,) for i in range(len(vectors))),
        phase_factor=factor,
    )


def _cube(cosine=Q(0), sine=Q(1)):
    # Vertex 2*j+b has the phase of C4 position j on either P2 layer b.
    base = ((Q(1), Q(0)), (cosine, sine), (Q(-1), Q(0)), (-cosine, -sine))
    vectors = tuple(vector for vector in base for _ in range(2))
    neighbors = tuple(
        (2 * j + 1 - b, 2 * ((j - 1) % 4) + b, 2 * ((j + 1) % 4) + b)
        for j in range(4)
        for b in range(2)
    )
    return vectors, neighbors


def _assert_cube_cancellation(vectors, neighbors):
    # The two horizontal antipodal phasors cancel; the vertical one remains.
    for vector, row in zip(vectors, neighbors):
        resultant = tuple(sum((vectors[j][axis] for j in row), Q(0)) for axis in (0, 1))
        assert resultant == vector
        assert sum((value * value for value in resultant), Q(0)) == 1


def test_connected_cube_at_quadrature_has_four_tangent_directions():
    vectors, neighbors = _cube()
    _assert_cube_cancellation(vectors, neighbors)
    reference = _derive(vectors, neighbors)
    result = observe_phase_source_geometry(reference)

    # R is a vertical swap, despite the connected degree-three cube support.
    expected = tuple(tuple(Q(j == (i ^ 1)) for j in range(8)) for i in range(8))
    assert reference.mean_response == expected
    assert reference.mean_resultant_squared == (1,) * 8
    assert result.reference == reference
    assert result.rank == 4
    assert result.tangent_dimension == 4
    assert result.mean_is_nonnegative
    assert not result.only_common_rotation
    assert result.scaled_source_jacobian == tuple(
        tuple(Q(j == (i ^ 1)) - Q(i == j) for j in range(8)) for i in range(8)
    )


def test_cube_antipodal_family_has_two_tangents_at_a_rational_oblique_point():
    vectors, neighbors = _cube(Q(3, 5), Q(4, 5))
    _assert_cube_cancellation(vectors, neighbors)
    reference = _derive(vectors, neighbors)
    result = observe_phase_source_geometry(reference)

    assert reference.mean_resultant_squared == (1,) * 8
    assert reference.mean_response[0] == (0, 1, Q(3, 5), 0, 0, 0, Q(-3, 5), 0)
    assert result.rank == 6
    assert result.tangent_dimension == 2
    assert not result.mean_is_nonnegative
    assert not result.only_common_rotation
    # Both common rotation and relative motion of the two antipodal pairs
    # lie in the exact tangent kernel. No finite trajectory is inferred.
    relative = (0, 0, 1, 1, 0, 0, 1, 1)
    for row in result.scaled_source_jacobian:
        assert sum(row) == 0
        assert sum(value * direction for value, direction in zip(row, relative)) == 0


def test_signed_k4_is_rigid_even_when_the_merged_stage_is_identity():
    vectors = ((Q(1), Q(0)),) * 3 + ((Q(-3, 5), Q(4, 5)),)
    neighbors = tuple(tuple(j for j in range(4) if j != i) for i in range(4))
    reference = _derive(vectors, neighbors, factor=Q(0))
    result = observe_phase_source_geometry(reference)

    assert reference.jacobian == tuple(
        tuple(Q(i == j) for j in range(4)) for i in range(4)
    )
    assert reference.is_nonnegative
    assert reference.mean_response[0] == (0, Q(7, 13), Q(7, 13), Q(-1, 13))
    assert reference.mean_response[3] == (Q(1, 3), Q(1, 3), Q(1, 3), 0)
    assert reference.mean_resultant_squared == (Q(13, 5), Q(13, 5), Q(13, 5), 9)
    assert result.rank == 3
    assert result.tangent_dimension == 1
    assert not result.mean_is_nonnegative
    assert result.only_common_rotation


def test_connected_acute_p3_has_only_common_rotation():
    reference = _derive(
        ((Q(1), Q(0)), (Q(3, 5), Q(4, 5)), (Q(0), Q(1))),
        ((1,), (0, 2), (1,)),
    )
    result = observe_phase_source_geometry(reference)

    assert reference.mean_response == ((0, 1, 0), (Q(1, 2), 0, Q(1, 2)), (0, 1, 0))
    assert result.rank == 2
    assert result.tangent_dimension == 1
    assert result.mean_is_nonnegative
    assert result.only_common_rotation


def test_disconnected_pairs_retain_independent_common_rotations():
    reference = _derive(((Q(1), Q(0)),) * 4, ((1,), (0,), (3,), (2,)))
    result = observe_phase_source_geometry(reference)

    assert result.rank == 2
    assert result.tangent_dimension == 2
    assert result.mean_is_nonnegative
    assert not result.only_common_rotation
    first_component = (1, 1, 0, 0)
    assert all(
        sum(value * direction for value, direction in zip(row, first_component)) == 0
        for row in result.scaled_source_jacobian
    )


def test_source_geometry_is_covariant_under_vertex_permutation():
    vectors, neighbors = _cube(Q(3, 5), Q(4, 5))
    original = observe_phase_source_geometry(_derive(vectors, neighbors))
    permutation = (3, 0, 7, 4, 1, 6, 2, 5)
    inverse = {old: new for new, old in enumerate(permutation)}
    permuted = observe_phase_source_geometry(
        _derive(
            tuple(vectors[old] for old in permutation),
            tuple(tuple(inverse[j] for j in neighbors[old]) for old in permutation),
        )
    )

    assert permuted.scaled_source_jacobian == tuple(
        tuple(original.scaled_source_jacobian[i][j] for j in permutation)
        for i in permutation
    )
    assert permuted.rank == original.rank
    assert permuted.tangent_dimension == original.tangent_dimension
    assert permuted.mean_is_nonnegative == original.mean_is_nonnegative
    assert permuted.only_common_rotation == original.only_common_rotation


@pytest.mark.parametrize("value", (None, {}, (), 1))
def test_source_geometry_requires_a_phase_response_reference(value):
    with pytest.raises(TypeError):
        observe_phase_source_geometry(value)


@pytest.mark.parametrize(
    "field",
    (
        "mean_response",
        "receiver_response",
        "jacobian",
        "mean_resultant_squared",
        "row_sum_residuals",
        "is_nonnegative",
    ),
)
def test_cached_reference_coefficients_are_rederived_before_use(field):
    reference = _derive(((Q(1), Q(0)),) * 2, ((1,), (0,)))
    if field in ("mean_response", "receiver_response", "jacobian"):
        replacement = ((Q(1), Q(0)), (Q(0), Q(1)))
    elif field == "mean_resultant_squared":
        replacement = (Q(2), Q(1))
    elif field == "row_sum_residuals":
        replacement = (Q(1), Q(0))
    else:
        replacement = False

    with pytest.raises(ValueError):
        observe_phase_source_geometry(replace(reference, **{field: replacement}))


def test_source_geometry_and_retained_reference_are_immutable():
    reference = _derive(((Q(1), Q(0)),) * 2, ((1,), (0,)))
    result = observe_phase_source_geometry(reference)

    with pytest.raises(FrozenInstanceError):
        result.rank = 0
    with pytest.raises(TypeError):
        result.scaled_source_jacobian[0][0] = Q(0)
    with pytest.raises(FrozenInstanceError):
        result.reference.phase_factor = Q(0)
    assert reference.mean_response == ((0, 1), (1, 0))
