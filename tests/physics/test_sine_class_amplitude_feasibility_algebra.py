"""Independent algebra for common-amplitude admission without new responses.

Low-order formal coefficients test homogeneity with carried events. Exact
incidence identities establish the heat-pressure spectral estimate; no heat
trajectory, reserved coefficient or complete nonlinear response is evaluated.
"""

from fractions import Fraction as Q
from math import factorial

import pytest

from tests.physics.test_sine_class_cubic_response_algebra import (
    _algebraic_edge_weights,
    _original_time_coefficients,
)
from tests.physics.test_sine_class_cubic_response_algebra import (
    geometry as _source_geometry,
)
from tnfr.mathematics._rational_interval import I
from tnfr.physics import relational_sine_class_cubic_response as cubic
from tnfr.physics.relational_sine_class_amplitude_feasibility import (
    _delayed_laplacian_bounds,
    bound_sine_class_amplitude_feasibility,
)


@pytest.fixture(scope="module")
def geometry():
    return _source_geometry.__wrapped__()


@pytest.fixture(autouse=True)
def no_reserved_calculation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("this algebra owner must not evaluate a response")

    monkeypatch.setattr(cubic, "_class_cubic_coefficients", forbidden)
    monkeypatch.setattr(cubic, "bound_sine_class_cubic_response", forbidden)


def _scale_levels(state, scale):
    return tuple(
        tuple(scale ** (row // 2 + 1) * value for value in values)
        for row, values in enumerate(state)
    )


def _polynomial_value(series, duration):
    return tuple(
        tuple(
            sum(
                duration**n * coefficient[row][node]
                for n, coefficient in enumerate(series)
            )
            for node in range(27)
        )
        for row in range(6)
    )


def _form_event(state, amplitude):
    return tuple(
        tuple(
            value + amplitude if row == 0 and node == 4 else value
            for node, value in enumerate(values)
        )
        for row, values in enumerate(state)
    )


@pytest.mark.parametrize("scale", (Q(7, 5), Q(-3, 2)))
def test_carried_variation_and_second_event_are_homogeneous(geometry, scale):
    # Arbitrary rational edge weights and short polynomial pieces are algebra
    # controls, not the retained design's trigonometry or finite coefficient.
    gamma, first, second, duration = Q(1, 32), Q(2, 13), Q(-3, 17), Q(1, 11)
    cosine, sine = _algebraic_edge_weights()
    zero = ((Q(0),) * 27,) * 6
    initial = _form_event(zero, first)
    prefix = _original_time_coefficients(
        geometry, initial, gamma, cosine, sine, order=4
    )
    scaled_prefix = _original_time_coefficients(
        geometry, _form_event(zero, scale * first), gamma, cosine, sine, order=4
    )
    assert scaled_prefix == tuple(_scale_levels(row, scale) for row in prefix)
    carried = _polynomial_value(prefix, duration)
    assert all(any(carried[row]) for row in (0, 1, 2, 3, 4))
    after = _form_event(carried, second)
    scaled_after = _form_event(
        _polynomial_value(scaled_prefix, duration), scale * second
    )
    assert scaled_after == _scale_levels(after, scale)
    assert after[1:] == carried[1:]

    suffix = _original_time_coefficients(geometry, after, gamma, cosine, sine, order=3)
    scaled_suffix = _original_time_coefficients(
        geometry, scaled_after, gamma, cosine, sine, order=3
    )
    assert scaled_suffix == tuple(_scale_levels(row, scale) for row in suffix)
    assert all(any(suffix[1][row]) for row in range(6))

    # Connect the independent original-coordinate recurrence to the actual
    # conditioned coefficient owner, including its held gamma scalings.
    _, degrees, _ = geometry
    parameters = cubic._CubicParameters(
        (1, 2, 1),
        I(gamma),
        I(gamma**2),
        degrees,
        tuple(map(I, sine)),
        tuple(map(I, cosine)),
    )
    units = (Q(1), gamma, gamma**3, gamma**4, gamma**4, gamma**5)
    conditioned = tuple(
        tuple(
            value / units[2 * level + channel]
            for channel in (0, 1)
            for value in carried[2 * level + channel]
        )
        for level in range(3)
    )
    production_after = cubic._event_levels(conditioned, second)
    actual = cubic._time_coefficients(production_after, parameters, order=3)
    for level in range(3):
        for n in range(4):
            for channel in (0, 1):
                row = 2 * level + channel
                for node, value in enumerate(suffix[n][row]):
                    bound = actual[level][n][27 * channel + node]
                    assert bound.contains(value / units[row])
                    assert bound.width < Q(1, 10**27)


def _laplacian(edges):
    matrix = [[Q(0)] * 27 for _ in range(27)]
    for i, j in edges:
        matrix[i][i] += 1
        matrix[j][j] += 1
        matrix[i][j] -= 1
        matrix[j][i] -= 1
    return tuple(map(tuple, matrix))


def _gram(rows):
    return tuple(
        tuple(sum(row[i] * row[j] for row in rows) for j in range(27))
        for i in range(27)
    )


def test_degree_metric_and_two_incidence_factorizations_bound_the_spectrum(geometry):
    edges, degrees, _ = geometry
    laplacian = _laplacian(edges)
    normalized = tuple(
        tuple(value / degree for value in row)
        for degree, row in zip(degrees, laplacian)
    )
    minus = tuple(tuple(Q(k == i) - Q(k == j) for k in range(27)) for i, j in edges)
    plus = tuple(tuple(Q(k == i) + Q(k == j) for k in range(27)) for i, j in edges)
    assert _gram(minus) == laplacian
    assert _gram(plus) == tuple(
        tuple(2 * degrees[i] * Q(i == j) - laplacian[i][j] for j in range(27))
        for i in range(27)
    )
    assert (
        tuple(
            tuple(degrees[i] * normalized[i][j] for j in range(27)) for i in range(27)
        )
        == laplacian
    )
    assert all(laplacian[i][j] == laplacian[j][i] for i in range(27) for j in range(27))
    assert all(sum(row) == 0 and sum(map(abs, row)) == 2 for row in normalized)
    assert degrees[4] == 3
    # The two Gram identities give 0 <= L <= 2D as quadratic forms.
    # Therefore D^-1/2 L D^-1/2 is symmetric with spectrum in [0,2].
    # Its diagonal spectral weights sum to one, and diagonal similarity
    # preserves [A exp(-A)]44; no heat response needs to be computed.


def test_positive_exponential_polynomial_certifies_the_heat_pressure_constant():
    # For z>=0 the omitted exponential terms are nonnegative. This exact
    # polynomial certificate proves z*exp(-z)<3/8, hence the donor bound
    # 3*[A exp(-A)]44<9/8 from the positive diagonal spectral measure.
    exponential_prefix = tuple(Q(1, factorial(n)) for n in range(5))
    difference = list(exponential_prefix)
    difference[1] -= Q(8, 3)
    square = (Q(1), Q(-2), Q(1))
    positive = (Q(23), Q(6), Q(1))  # (z+3)^2+14
    product = [Q(0)] * 5
    for i, a in enumerate(square):
        for j, b in enumerate(positive):
            product[i + j] += a * b
    product[0] += 1
    assert tuple(difference) == tuple(value / 24 for value in product)
    assert sum(exponential_prefix) > Q(8, 3)
    assert Q(3) / sum(exponential_prefix) < Q(9, 8)


@pytest.mark.parametrize("pulse", (Q(7, 10000), Q(-3, 8000)))
def test_exact_form_jump_retains_carried_pressure_and_weighted_charge(geometry, pulse):
    edges, degrees, _ = geometry
    state = tuple(Q(((5 * i + 3) % 11) - 5, 10000) for i in range(27))
    after = tuple(value + pulse if i == 4 else value for i, value in enumerate(state))

    def kinetic(values):
        return sum((values[i] - values[j]) ** 2 for i, j in edges) / 2

    pressure = sum(
        state[4] - state[j if i == 4 else i] for i, j in edges if 4 in (i, j)
    )
    work = kinetic(after) - kinetic(state)
    assert work == pulse * pressure + Q(3, 2) * pulse**2
    assert pulse * pressure != 0
    assert sum(d * (v - u) for d, v, u in zip(degrees, after, state)) == 3 * pulse
    assert all(after[i] == state[i] for i in range(27) if i != 4)


def test_heat_defect_pressure_bound_uses_the_actual_donor_row_norm(geometry):
    laplacian = _laplacian(geometry[0])
    row = laplacian[4]
    assert sum(map(abs, row)) == 6
    g, eps, amplitude = Q(1, 3000), Q(1, 10**32), Q(7, 10000)
    q = (amplitude + eps + 2 * g * eps) / (1 - 2 * g**2)
    defect = eps + 2 * g * eps + 2 * g**2 * q
    maximizing_error = tuple(
        defect if value > 0 else -defect if value < 0 else Q(0) for value in row
    )
    assert (
        sum(value * error for value, error in zip(row, maximizing_error)) == 6 * defect
    )
    # Heat pressure lies in [0,9a/8]. Both endpoints of its Minkowski sum
    # with the full-law/source row error are needed; positivity of the
    # nominal heat pressure does not make the actual pressure nonnegative.
    lower, upper = -6 * defect, Q(9, 8) * amplitude + 6 * defect
    assert lower < 0 < upper
    assert upper < Q(79, 10**5)
    assert _delayed_laplacian_bounds(amplitude) == (lower, upper)


def test_whole_scale_interval_has_independent_noise_work_and_identity_budgets():
    scale_lower, scale_upper = Q(4, 3), Q(7, 5)
    g, eps, delta, radius = Q(1, 3000), Q(1, 10**32), Q(1, 10**30), Q(1, 12)
    pulse, time = scale_upper / 2000, Q(2)
    q_before = (pulse + eps + 2 * g * eps) / (1 - 2 * g**2)
    defect = eps + 2 * g * eps + 2 * g**2 * q_before
    work_first = Q(3, 2) * pulse**2 + 6 * pulse * eps
    work_second = Q(3, 2) * pulse**2 + pulse * (Q(9, 8) * pulse + 6 * defect)
    assert work_first < Q(2, 10**6)
    assert work_second < Q(2, 10**6)
    assert 8 * eps**2 < Q(1, 10**12)
    # No continuous-loss credit: bound storage using the two complete
    # event works, including the carried pressure at the delayed event.
    assert 22 * eps**2 + work_first + work_second < radius**2 / 2700 == Q(1, 388800)
    q_all = (2 * pulse + eps + 2 * g * time * eps) / (1 - 2 * g**2 * time**2)
    phase = eps + 2 * g * time * q_all
    assert 27 * (q_all**2 + phase**2) < radius**2

    def tail(amplitude):
        assert 2 * g * amplitude < 1
        return (
            256
            * g**6
            * amplitude**5
            * time
            / ((1 - 2 * g**2 * time**2) * (1 - 4 * g**2 * amplitude**2))
        )

    assert 16 * g**2 * time**2 < 1
    error = 2 * (tail(2 * pulse) + 2 * tail(pulse)) + 8 * eps / (1 - 2 * g * time)
    # This is a declared coefficient premise, deliberately rounded beyond
    # the retained enclosure, not a fresh coefficient calculation.
    coefficient_magnitude_lower = Q(7013, 10**33)
    oriented_lower = scale_lower**3 * coefficient_magnitude_lower - error
    assert oriented_lower > 16 * delta > 8 * delta
    # Every error/work expression used above increases with a nonnegative
    # pulse; the signal floor uses the lower scale. These opposite endpoints
    # certify the whole interval, without sampling candidate interventions.
    report = bound_sine_class_amplitude_feasibility(
        amplitude_scale_lower=scale_lower,
        amplitude_scale_upper=scale_upper,
        base_cubic_lower=-Q(7014, 10**33),
        base_cubic_upper=-coefficient_magnitude_lower,
    )
    assert report.feasible_interval_certified
    both = report.upper_scale_history_bounds[3]
    assert both.first_probe_work_upper_bound == work_first
    assert both.second_probe_work_upper_bound == work_second
    assert both.first_probe_work_bounds.contains(work_first)
    assert both.second_probe_work_bounds.contains(work_second)
    assert (
        both.after_second_excess_storage_upper_bound
        == 22 * eps**2 + work_first + work_second
    )
    assert both.after_second_radius_squared_upper_bound == 27 * (q_all**2 + phase**2)
    # The owner uses a tighter outward upper bound for gamma in the source
    # allowance. All Cauchy and event bounds use the same rational g as above.
    assert report.decision.oriented_lower >= oriented_lower
