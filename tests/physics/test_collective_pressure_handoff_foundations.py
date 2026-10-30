"""Synthetic preflight controls for the simultaneous IL work argument.

These edge identities and uniform algebraic bounds precede the maintained
contact IL/OZ study. They execute no operator, source producer or trajectory.
They neither evaluate its reserved response nor turn pressure telemetry into
an independent dynamical coordinate.
"""

from fractions import Fraction as Q

import pytest

from tnfr.mathematics._exact_linear_algebra import exact_symmetric_semidefinite
from tnfr.mathematics._rational_interval import I, atan, cos, pi_interval, sin, sqrt


def _phase_energy(phases, edges):
    """Sum every undirected edge exactly once, independently of target rows."""
    return sum((1 - cos(I(phases[j] - phases[i])) for i, j in edges), I(0))


@pytest.mark.parametrize("shared_edge", (False, True))
def test_simultaneous_work_keeps_the_shared_edge_correction(shared_edge):
    phases = (Q(0), Q(1, 16), Q(1, 8), Q(-1, 8), Q(1, 12))
    edges = ((0, 2), (0, 3), (1, 2), (1, 4), (3, 4))
    if shared_edge:
        edges += ((0, 1),)
    changes = (Q(1, 32), Q(-1, 48))
    after = tuple(
        value + (changes[i] if i < 2 else 0) for i, value in enumerate(phases)
    )
    full_work = _phase_energy(after, edges) - _phase_energy(phases, edges)

    local_work = I(0)
    for target, change in enumerate(changes):
        neighbors = tuple(j if i == target else i for i, j in edges if target in (i, j))
        local_work += sum(
            (
                cos(I(phases[node] - phases[target]))
                - cos(I(phases[node] - phases[target] - change))
                for node in neighbors
            ),
            I(0),
        )

    left, right = changes
    correction = (
        -4
        * cos(I(phases[1] - phases[0] + (right - left) / 2))
        * sin(I(left / 2))
        * sin(I(right / 2))
        if shared_edge
        else I(0)
    )
    residual = full_work - local_work - correction
    assert residual.contains(0) and residual.width < Q(1, 2**100)
    if shared_edge:
        assert not (full_work - local_work).contains(0)
        assert correction.abs_max < (left**2 + right**2) / 2


def test_full_swap_invalidates_addition_of_single_target_energy_changes():
    phases, edges = (Q(0), Q(1, 2)), ((0, 1),)
    before = _phase_energy(phases, edges)
    simultaneous = _phase_energy(tuple(reversed(phases)), edges) - before
    independent = (
        _phase_energy((phases[1], phases[1]), edges)
        + _phase_energy((phases[0], phases[0]), edges)
        - 2 * before
    )
    correction = 4 * sin(I((phases[1] - phases[0]) / 2)) ** 2
    assert simultaneous.contains(0)
    assert simultaneous.width < Q(1, 2**100)
    assert independent.hi < 0 < correction.lo
    assert (simultaneous - independent - correction).contains(0)


def test_degree_seven_resultant_bound_keeps_the_entire_identity_ball():
    alpha = 2 * pi_interval() / 5
    # One twin, four old-cycle neighbors and two aligned opposite members.
    offsets = (I(0), alpha, alpha, -alpha, -alpha, I(0), I(0))
    real = sum((cos(angle) for angle in offsets), I(0))
    imaginary = sum((sin(angle) for angle in offsets), I(0))
    assert (real - (2 + sqrt(I(5)))).contains(0)
    assert imaginary.contains(0)

    # Cauchy controls all seven phase errors jointly, not seven separate radii.
    degree = len(offsets)
    cauchy = tuple(
        tuple(Q(degree * (i == j) - 1) for j in range(degree)) for i in range(degree)
    )
    assert exact_symmetric_semidefinite(cauchy)
    assert not exact_symmetric_semidefinite(cauchy, strict=True)
    error = sqrt(I(degree)) / 16
    assert error.hi < Q(1, 6)
    real_lower = real.lo - error.hi
    assert real_lower > 4
    direction = atan(I(error.hi / real_lower))
    assert direction.hi < Q(1, 24)
    assert Q(1, 16) + direction.hi < Q(5, 48) < Q(1, 8)


def test_passivity_polynomial_is_nonnegative_for_every_phase_coefficient():
    # On |a|<=1/8, the sine Taylor remainder gives sin(t)/t>=3/4.
    radius = Q(1, 8)
    sine_floor = Q(1) - radius**2 / 6
    assert sine_floor > Q(3, 4)
    resultant_lower = Q(4)
    integral_factor = resultant_lower * Q(3, 4) / 2

    # -DeltaV-||d||^2 is bounded below by a quadratic in a=(a_u,a_v).
    # For either sign of the shared-edge cross term, its matrix polynomial is
    # lambda*(1-lambda)*linear + lambda**2*endpoint. Both matrices are PSD.
    for sign in (-1, 1):
        linear = tuple(
            tuple(2 * integral_factor * (i == j) for j in range(2)) for i in range(2)
        )
        quadratic = tuple(
            tuple(-(integral_factor + 1) if i == j else -Q(sign, 2) for j in range(2))
            for i in range(2)
        )
        endpoint = tuple(
            tuple(linear[i][j] + quadratic[i][j] for j in range(2)) for i in range(2)
        )
        assert exact_symmetric_semidefinite(linear, strict=True)
        assert exact_symmetric_semidefinite(endpoint)
        assert not exact_symmetric_semidefinite(endpoint, strict=True)

    # Fixed abstract local-angle probes check the nonlinear trigonometric
    # expression against that uniform certificate; these are not graph data.
    angles = (Q(1, 8), Q(-1, 16))
    gap = Q(1, 12)
    for coefficient in (Q(1, 16), Q(3, 10), Q(1)):
        changes = tuple(coefficient * angle for angle in angles)
        local = sum(
            (
                resultant_lower * (cos(I(angle)) - cos(I((1 - coefficient) * angle)))
                for angle in angles
            ),
            I(0),
        )
        left, right = changes
        correction = (
            -4 * cos(I(gap + (right - left) / 2)) * sin(I(left / 2)) * sin(I(right / 2))
        )
        assert (local + correction).hi < -sum(change**2 for change in changes)
