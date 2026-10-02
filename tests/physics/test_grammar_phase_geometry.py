"""Exact strict-U3 controls for a nonzero canonical phase source.

The connected double-star has a singular source derivative while its local
fixed-source fiber is only common rotation. No phase evolution is selected.
"""

from fractions import Fraction as Q

from tnfr.physics.phase_response import (
    derive_phase_response,
    observe_phase_source_geometry,
)

# Order: u, v, left leaves a,b, right leaves c,d. An entry (a,b) denotes
# a + i*sqrt(3)*b; this permits exact root-of-unity arithmetic over rationals.
_PHASORS = (
    (Q(1), Q(0)),
    (Q(1, 2), Q(1, 2)),
    (Q(1, 2), Q(-1, 2)),
    (Q(1, 2), Q(-1, 2)),
    (Q(-1, 2), Q(1, 2)),
    (Q(-1, 2), Q(1, 2)),
)
_NEIGHBORS = ((1, 2, 3), (0, 4, 5), (0,), (0,), (1,), (1,))
_PHASE_PI = (Q(0), Q(1, 3), Q(-1, 3), Q(-1, 3), Q(2, 3), Q(2, 3))


def _multiply(left, right):
    a, b = left
    c, d = right
    return a * c - 3 * b * d, a * d + b * c


def _conjugate(value):
    return value[0], -value[1]


def _norm_squared(value):
    a, b = value
    return a * a + 3 * b * b


def _divide(left, right):
    denominator = _norm_squared(right)
    return tuple(value / denominator for value in _multiply(left, _conjugate(right)))


def _resultants():
    return tuple(
        tuple(sum((_PHASORS[j][axis] for j in row), Q(0)) for axis in (0, 1))
        for row in _NEIGHBORS
    )


def _reference():
    gram = tuple(
        tuple(_multiply(left, _conjugate(right))[0] for right in _PHASORS)
        for left in _PHASORS
    )
    return derive_phase_response(
        cosine_gram=gram,
        mean_neighbors=_NEIGHBORS,
        receiver_sources=tuple((i,) for i in range(6)),
        phase_factor=Q(1, 2),
    )


def test_double_star_has_strict_u3_edges_and_regular_nonzero_source():
    reference = _reference()
    resultants = _resultants()
    assert all(_norm_squared(z) == 1 for z in _PHASORS)
    assert tuple(map(_norm_squared, resultants)) == (3, 3, 1, 1, 1, 1)
    assert reference.mean_resultant_squared == (3, 3, 1, 1, 1, 1)
    for i, row in enumerate(_NEIGHBORS):
        for j in row:
            gap = abs((_PHASE_PI[j] - _PHASE_PI[i] + 1) % 2 - 1)
            assert gap == Q(1, 3) < Q(1, 2)
            assert reference.cosine_gram[i][j] == Q(1, 2)

    centered = tuple(_divide(s, z) for s, z in zip(resultants, _PHASORS))
    assert centered == (
        (Q(3, 2), Q(-1, 2)),
        (Q(3, 2), Q(1, 2)),
        (Q(1, 2), Q(1, 2)),
        (Q(1, 2), Q(1, 2)),
        (Q(1, 2), Q(-1, 2)),
        (Q(1, 2), Q(-1, 2)),
    )
    # Positive real parts fix the regular branch. Squared cosine plus sine
    # sign identifies source offsets (-pi/6,+pi/6,+pi/3,+pi/3,-pi/3,-pi/3).
    assert tuple(a * a / _norm_squared(z) for z in centered for a in (z[0],)) == (
        Q(3, 4),
        Q(3, 4),
        Q(1, 4),
        Q(1, 4),
        Q(1, 4),
        Q(1, 4),
    )
    assert tuple(1 if b > 0 else -1 for _, b in centered) == (-1, 1, 1, 1, -1, -1)


def test_strict_u3_does_not_make_nonzero_source_derivative_rigid():
    reference = _reference()
    assert reference.mean_response == (
        (0, 0, Q(1, 2), Q(1, 2), 0, 0),
        (0, 0, 0, 0, Q(1, 2), Q(1, 2)),
        (1, 0, 0, 0, 0, 0),
        (1, 0, 0, 0, 0, 0),
        (0, 1, 0, 0, 0, 0),
        (0, 1, 0, 0, 0, 0),
    )
    geometry = observe_phase_source_geometry(reference)
    assert geometry.rank == 4
    assert geometry.tangent_dimension == 2
    assert geometry.mean_is_nonnegative
    assert not geometry.only_common_rotation
    # The connected support bridge has zero differential influence in both
    # directions. The two component indicators are independent kernel vectors.
    for vector in ((1, 0, 1, 1, 0, 0), (0, 1, 0, 0, 1, 1)):
        assert all(
            sum(a * b for a, b in zip(row, vector)) == 0
            for row in geometry.scaled_source_jacobian
        )


def test_extra_tangent_changes_source_at_second_order():
    # Move v,c,d together by eta, holding u,a,b. In each center's rotating
    # frame, S(eta)=exp(+/-i*eta)*z_cross+2*z_leaf. If r=z_cross/S(0),
    # Arg(S)''=Im(-r+r*r), independent of the sign of eta. The field below
    # multiplies this derivative by sqrt(3), equivalently pi*sqrt(3)*g''.
    resultants = _resultants()
    center_ratios = (
        _divide(_PHASORS[1], resultants[0]),
        _divide(_PHASORS[0], resultants[1]),
    )
    assert center_ratios == ((Q(0), Q(1, 3)), (Q(0), Q(-1, 3)))
    scaled_curvature = tuple(3 * (_multiply(r, r)[1] - r[1]) for r in center_ratios)
    assert scaled_curvature == (-1, 1)
    # Nonzero curvature rejects this tangent line as a constant-source path.
    # The analytic fixed-leaf constraint further gives sin(delta+pi/6)=1,
    # so the nearby exact fiber has only common rotation despite nullity two.
