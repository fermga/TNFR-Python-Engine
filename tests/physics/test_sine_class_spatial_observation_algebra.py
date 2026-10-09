"""Independent symmetry and locality admission for a spatial receiver readout.

Exact low-order formal algebra admits the observation before its finite
coefficient controls. No nonlinear response or acquired-source assessment is
executed. Floating triangular integration is only an implementation control.
"""

from fractions import Fraction as Q

import numpy as np
import pytest

from tests.physics.test_sine_class_cubic_response_algebra import (
    _floating_variational_coefficient,
)
from tnfr.physics.relational_sine_class_spatial_observation import (
    bound_sine_class_spatial_observation,
)


@pytest.fixture(scope="module")
def report():
    # This is the declared projection after its first retained assessment.
    # Shared immutable class coefficients are reused across test modules;
    # source, observation and event consequences remain fresh calculations.
    return bound_sine_class_spatial_observation(
        first_probe_amplitude=Q(1, 2000),
        second_probe_amplitude=Q(1, 2000),
        delay=Q(1),
        total_duration=Q(2),
        endpoint_radius=Q(1, 10**32),
        readout_error_bound=Q(1, 10**30),
        radius=Q(1, 12),
        contact_work_allowance=Q(1, 10**12),
        first_probe_work_allowance=Q(2, 10**6),
        second_probe_work_allowance=Q(2, 10**6),
    )


@pytest.fixture(scope="module")
def geometry():
    edges = tuple(
        (9 * c + j, 9 * c + (j + 1) % 9) for c in range(3) for j in range(9)
    ) + ((4, 13), (13, 22))
    degrees = tuple(sum(i in edge for edge in edges) for i in range(27))
    reflection = tuple(9 * c + 8 - j for c in range(3) for j in range(9))
    laplacian = [[Q(0)] * 27 for _ in range(27)]
    for left, right in edges:
        for i, j in ((left, right), (right, left)):
            laplacian[i][i] += Q(1, degrees[i])
            laplacian[i][j] -= Q(1, degrees[i])
    return edges, degrees, reflection, tuple(map(tuple, laplacian))


def _receiver_odd(vector, reflection):
    return tuple(
        (vector[i] - vector[reflection[i]]) / 2 if i >= 18 else Q(0) for i in range(27)
    )


def _mv(matrix, vector):
    return tuple(sum(a * x for a, x in zip(row, vector)) for row in matrix)


def _weighted_laplacian(edges, degrees, weights):
    matrix = [[Q(0)] * 27 for _ in range(27)]
    for (left, right), weight in zip(edges, weights):
        for i, j in ((left, right), (right, left)):
            matrix[i][i] += weight / degrees[i]
            matrix[i][j] -= weight / degrees[i]
    return tuple(map(tuple, matrix))


def test_receiver_observer_selects_even_amplitude_and_rejects_central_parity(geometry):
    _, degrees, reflection, _ = geometry
    assert (degrees[4], degrees[13], degrees[22]) == (3, 4, 3)
    weights = tuple(Q(i == 23) - Q(i == 21) for i in range(27))
    assert sum(map(abs, weights)) == 2
    assert tuple(weights[reflection[i]] for i in range(27)) == tuple(
        -value for value in weights
    )
    # x_n has reflection parity (-1)^(n+1). Applying this readout therefore
    # annihilates odd amplitude degrees, whereas the central readout does
    # the reverse. This does not impose parity on actual source residuals.
    for amplitude_degree in range(1, 7):
        sign = (-1) ** (amplitude_degree + 1)
        values = tuple(Q(i % 9 - 4) for i in range(27))
        projected = tuple(
            (values[i] + sign * values[reflection[i]]) / 2 for i in range(27)
        )
        observation = projected[23] - projected[21]
        assert (observation != 0) == (amplitude_degree % 2 == 0)
        if amplitude_degree % 2 == 0:
            assert projected[22] == 0


def test_receiver_odd_sector_is_reducing_and_has_class_independent_linear_law(geometry):
    edges, degrees, reflection, laplacian = geometry
    matrices = tuple(
        _weighted_laplacian(
            edges,
            degrees,
            (Q(3, 4),) * 9 + (middle,) * 9 + (Q(3, 4),) * 9 + (Q(1),) * 2,
        )
        for middle in (Q(2, 3), Q(1, 5))
    )
    # Every basis column is checked, including forcing from the mediator and
    # the articulation port, rather than only one specially chosen state.
    for node in range(27):
        basis = tuple(Q(i == node) for i in range(27))
        projected = _receiver_odd(basis, reflection)
        for matrix in (laplacian, *matrices):
            assert _receiver_odd(_mv(matrix, basis), reflection) == _mv(
                matrix, projected
            )
        assert _mv(matrices[0], projected) == _mv(matrices[1], projected)
        if node < 18 or node == 22:
            assert not any(projected)


def test_eta_zero_quadratic_receiver_forcing_is_class_blind(geometry):
    edges, degrees, reflection, _ = geometry
    phase = tuple(Q((i % 9 - 4) ** 2 * (i // 9 + 1), 32) for i in range(27))
    forces = []
    for mediator_sine in (Q(4, 5), Q(3, 7)):
        sine = (Q(2, 3),) * 9 + (mediator_sine,) * 9 + (Q(2, 3),) * 9 + (Q(0),) * 2
        force = [Q(0)] * 27
        for (left, right), weight in zip(edges, sine):
            current = -weight * (phase[right] - phase[left]) ** 2 / 2
            force[left] += current / degrees[left]
            force[right] -= current / degrees[right]
        forces.append(tuple(force))
    assert forces[0] != forces[1]
    assert _receiver_odd(forces[0], reflection) == _receiver_odd(forces[1], reflection)
    difference = tuple(a - b for a, b in zip(*forces))
    assert all(difference[i] == 0 for i in (*range(9), *range(18, 27)))
    assert difference[13] == 0
    assert tuple(difference[reflection[i]] for i in range(27)) == tuple(
        -value for value in difference
    )


def _add(*polynomials):
    result = {}
    for polynomial in polynomials:
        for degree, coefficient in polynomial.items():
            result[degree] = result.get(degree, Q(0)) + coefficient
    return {
        degree: coefficient for degree, coefficient in result.items() if coefficient
    }


def _scale(polynomial, scalar):
    return {
        degree: scalar * value for degree, value in polynomial.items() if scalar * value
    }


def _multiply(left, right):
    terms = ({i + j: a * b} for i, a in left.items() for j, b in right.items())
    return _add(*terms)


def _eta(polynomial):
    return {degree + 1: value for degree, value in polynomial.items()}


def _quadratic_formal_series(geometry, mediator_cosine, mediator_sine, order=6):
    """Original edge algebra with symbolic eta; no time value is supplied."""
    edges, degrees, _, _ = geometry
    cosine = (Q(3, 4),) * 9 + (mediator_cosine,) * 9 + (Q(3, 4),) * 9 + (Q(1),) * 2
    sine = (Q(2, 3),) * 9 + (mediator_sine,) * 9 + (Q(2, 3),) * 9 + (Q(0),) * 2
    initial = tuple(
        tuple({0: Q(1)} if row == 0 and node == 4 else {} for node in range(27))
        for row in range(4)
    )
    coefficients = [initial]
    for n in range(order):
        rates = [[{} for _ in range(27)] for _ in range(4)]
        for edge, c, sn in zip(edges, cosine, sine):
            left, right = edge
            differences = tuple(
                tuple(_add(row[right], _scale(row[left], -1)) for row in coefficient)
                for coefficient in coefficients
            )
            square = _add(
                *(
                    _multiply(differences[p][1], differences[n - p][1])
                    for p in range(n + 1)
                )
            )
            for level in range(2):
                form = _add(
                    differences[n][2 * level],
                    _scale(_eta(differences[n][2 * level + 1]), c),
                    _scale(square, -sn / 2) if level == 1 else {},
                )
                phase = _scale(differences[n][2 * level], -1)
                for row, current in ((2 * level, form), (2 * level + 1, phase)):
                    rates[row][left] = _add(
                        rates[row][left], _scale(current, Q(1, degrees[left]))
                    )
                    rates[row][right] = _add(
                        rates[row][right], _scale(current, -Q(1, degrees[right]))
                    )
        coefficients.append(
            tuple(tuple(_scale(value, Q(1, n + 1)) for value in row) for row in rates)
        )
    return tuple(coefficients)


def test_first_class_sensitive_quadratic_time_coefficient_is_eta_order_one(geometry):
    c1, c2, receiver_sine = Q(2, 3), Q(1, 5), Q(2, 3)
    first = _quadratic_formal_series(geometry, c1, Q(4, 5))
    second = _quadratic_formal_series(geometry, c2, Q(3, 7))
    difference = []
    for left, right in zip(first, second):
        difference.append(
            _add(
                left[2][23],
                _scale(left[2][21], -1),
                _scale(right[2][23], -1),
                right[2][21],
            )
        )
    assert all(not polynomial for polynomial in difference[:6])
    assert difference[6] == {1: receiver_sine * (c1 - c2) / 20736}
    # The physical quadratic coefficient multiplies this scaled hierarchy by
    # gamma^3; the first mediator-sensitive term therefore has gamma^5.
    center_first = _add(first[3][1][22], _scale(second[3][1][22], -1))
    assert center_first == {1: -(c1 - c2) / 144}
    assert first[2][1][22] == {0: -Q(1, 24)}


def test_mixed_joint_time_polynomial_and_equal_time_limit():
    # Integrate t^2(t-s)^3+t^3(t-s)^2 after t=s+v. The result is homogeneous
    # of time degree six; it is not a sign prediction at the finite horizon.
    s, u = Q(3, 7), Q(5, 11)
    integrand = (Q(0), Q(0), s**3, 4 * s**2, 5 * s, Q(2))
    integral = sum(value * u ** (n + 1) / (n + 1) for n, value in enumerate(integrand))
    expected = s**3 * u**3 / 3 + s**2 * u**4 + s * u**5 + u**6 / 3
    assert integral == expected > 0
    # When s=0, the mixed coefficient of (a+b)^2 is twice the single-input
    # a^2 coefficient. This checks both the time integral and 1/3456 factor.
    assert Q(1, 3456) * Q(1, 3) == 2 * Q(1, 20736)


def test_class_sensitive_cauchy_tail_and_separate_source_noise_counts():
    g, time, a = Q(1, 3000), Q(2), Q(1, 2000)
    assert 16 * g**2 * time**2 < 1

    def tail(amplitude):
        disk = 1 / (2 * g * amplitude)
        assert disk > 1
        # Each full class sine field has norm <=8 on the disk. The common
        # events cancel in their difference; integrate both the phase row
        # and the receiver-local forcing before applying the two-node readout.
        form_difference_rate = 2 * 8 * g
        phase_difference_quadratic = 2 * g * form_difference_rate / 2
        receiver_odd_cubic = 8 * g * phase_difference_quadratic / 3
        boundary = 2 * receiver_odd_cubic * time**3
        assert boundary == Q(256, 3) * g**3 * time**3
        geometric = boundary * disk**-4 / (1 - disk**-2)
        compact = (
            Q(4096, 3) * g**7 * amplitude**4 * time**3 / (1 - 4 * g**2 * amplitude**2)
        )
        assert geometric == compact
        return compact

    # These are already cross-class remainders, so only the three nonzero
    # histories are summed, with no further factor two.
    remainder = tail(2 * a) + 2 * tail(a)
    assert remainder < Q(6, 10**33)
    source = 16 * Q(1, 10**32) / (1 - 2 * g * time)
    assert source + remainder < Q(17, 10**32)
    noise = Q(1, 10**30)
    signs = (1, -1, -1, 1, -1, 1, 1, -1)
    raw_signs = tuple(value for sign in signs for value in (sign, -sign))
    assert len(raw_signs) == sum(map(abs, raw_signs)) == 16
    actual = 15 * noise
    errors = tuple(-sign * actual / 16 for sign in raw_signs)
    assert max(map(abs, errors)) <= noise
    assert actual + sum(sign * error for sign, error in zip(raw_signs, errors)) == 0
    # A separately noisy null statistic carries another 16-error allowance;
    # true sign, recorded sign and null exclusion are distinct obligations.
    assert 17 * noise - 16 * noise > 0
    assert 17 * noise - 32 * noise < 0
    assert 33 * noise - 32 * noise > 0


def test_production_higher_even_bound_keeps_the_class_difference_scope():
    from tnfr.physics.relational_sine_class_spatial_observation import (
        _higher_even_contrast_remainder,
    )

    # A formula-only call, not a finite coefficient or response assessment.
    amplitude, time, g = Q(3, 7000), Q(7, 5), Q(1, 3000)
    rho = 1 / (2 * g * amplitude)
    disk_bound = Q(256, 3) * g**3 * time**3
    expected = disk_bound * rho**-4 / (1 - rho**-2)
    assert _higher_even_contrast_remainder(amplitude, time) == expected
    assert _higher_even_contrast_remainder(Q(0), time) == 0


def test_finite_quadratic_projection_matches_independent_triangular_integration(
    geometry, report
):
    independently_integrated = tuple(
        _floating_variational_coefficient(
            geometry[:3],
            mediator_class,
            amplitude_level=1,
            weights=((23, 1), (21, -1)),
        )
        for mediator_class in (1, 2)
    )
    retained = tuple(
        float((lower + upper) / 2)
        for lower, upper in report.scaled_class_quadratic_bounds
    )
    assert min(map(abs, independently_integrated)) > 1e-25
    assert np.allclose(independently_integrated, retained, rtol=1e-8, atol=0)
    difference = independently_integrated[0] - independently_integrated[1]
    lower, upper = report.complete_quadratic_contrast_bounds
    assert Q(7, 10**30) < difference < Q(9, 10**30)
    assert np.isclose(difference, float((lower + upper) / 2), rtol=1e-6, atol=0)


def test_true_spatial_sign_and_simultaneous_scalar_cancellation(report):
    from tnfr.physics.relational_sine_class_cubic_response import (
        bound_sine_class_cubic_response,
    )

    names = (
        "first_probe_amplitude",
        "second_probe_amplitude",
        "delay",
        "total_duration",
        "endpoint_radius",
        "readout_error_bound",
        "radius",
        "contact_work_allowance",
        "first_probe_work_allowance",
        "second_probe_work_allowance",
    )
    central = bound_sine_class_cubic_response(
        **{name: getattr(report, name) for name in names}
    )
    spatial_bounds = report.decision.true_bounds
    assert 0 < spatial_bounds[0] < spatial_bounds[1] < 16 * report.readout_error_bound
    assert max(map(abs, central.decision.true_bounds)) < 8 * report.readout_error_bound
    # Check all endpoint pairs of the two certified marginal rectangles. The
    # construction is affine, so it holds throughout and hence for every
    # admissible actual pair, without requiring marginal attainability.
    history_signs = (1, -1, -1, 1, -1, 1, 1, -1)
    for dc in central.decision.true_bounds:
        for ds in spatial_bounds:
            central_errors = tuple(-sign * dc / 8 for sign in history_signs)
            high_errors = tuple(-sign * ds / 16 for sign in history_signs)
            low_errors = tuple(sign * ds / 16 for sign in history_signs)
            all_errors = central_errors + high_errors + low_errors
            assert max(map(abs, all_errors)) <= report.readout_error_bound
            assert dc + sum(s * e for s, e in zip(history_signs, central_errors)) == 0
            assert (
                ds
                + sum(
                    s * (hi - lo)
                    for s, hi, lo in zip(history_signs, high_errors, low_errors)
                )
                == 0
            )
    # The raw channels are disjoint (nodes22,23,21), so one allowed error
    # assignment erases both mixed statistics simultaneously. This does not
    # assert overlap of their complete raw endpoint-record vectors.
