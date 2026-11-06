"""Independent algebra for organization-dependent nonlinear interaction.

Exact graph and observation algebra, plus an auxiliary heat integral, check
the conditional contrast. No complete sine trajectory, acquired-source
assessment, retained producer or archived worker is executed.
"""

from fractions import Fraction as Q
from itertools import product
from math import factorial

import numpy as np
import pytest

from tnfr.physics.relational_sine_class_nonlinear_organization import (
    bound_sine_class_nonlinear_organization,
)


@pytest.fixture(scope="module")
def report():
    return bound_sine_class_nonlinear_organization(
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
    cycles = tuple(
        tuple((9 * part + j, 9 * part + (j + 1) % 9) for j in range(9))
        for part in range(3)
    )
    edges = sum(cycles, ()) + ((4, 13), (13, 22))
    degrees = tuple(sum(node in edge for edge in edges) for node in range(27))
    laplacian = [[Q(0) for _ in range(27)] for _ in range(27)]
    for left, right in edges:
        for node, neighbor in ((left, right), (right, left)):
            laplacian[node][node] += 1
            laplacian[node][neighbor] -= 1
    normalized = tuple(
        tuple(value / degrees[node] for value in row)
        for node, row in enumerate(laplacian)
    )
    return cycles, edges, degrees, normalized


def _mv(matrix, vector):
    return tuple(sum(a * b for a, b in zip(row, vector)) for row in matrix)


def _edge_cubic(edges, degrees, first, second, edge_weights):
    """Direct signed edge currents, independently of the heat-polynomial code."""
    values = [Q(0)] * 27
    for (left, right), weight in zip(edges, edge_weights):
        p = first[right] - first[left]
        q = second[right] - second[left]
        current = weight * p * q * (p + q) / 2
        values[left] -= current / degrees[left]
        values[right] += current / degrees[right]
    return tuple(values)


def test_class_difference_is_exactly_the_selected_mediator_edge_operator(geometry):
    cycles, edges, degrees, _ = geometry
    first = tuple(Q((7 * i) % 13 - 6, 17) for i in range(27))
    second = tuple(Q((5 * i) % 11 - 5, 19) for i in range(27))
    outer, c1, c2 = Q(3, 4), Q(2, 3), Q(1, 5)
    selected = tuple(Q(edge in cycles[1]) for edge in edges)
    weights1 = tuple(
        c1 if edge in cycles[1] else outer if edge in sum(cycles, ()) else Q(1)
        for edge in edges
    )
    weights2 = tuple(
        c2 if edge in cycles[1] else outer if edge in sum(cycles, ()) else Q(1)
        for edge in edges
    )
    f1 = _edge_cubic(edges, degrees, first, second, weights1)
    f2 = _edge_cubic(edges, degrees, first, second, weights2)
    mediator = _edge_cubic(edges, degrees, first, second, selected)
    assert any(mediator)
    assert tuple(a - b for a, b in zip(f1, f2)) == tuple(
        (c1 - c2) * value for value in mediator
    )
    assert sum(d * value for d, value in zip(degrees, mediator)) == 0
    assert all(mediator[node] == 0 for node in (*range(9), *range(18, 27)))


def test_mediator_heat_onset_is_degree_five_not_the_common_quartic_term(geometry):
    cycles, edges, degrees, normalized = geometry
    impulse = tuple(Q(i == 4) for i in range(27))
    direction = _mv(normalized, impulse)
    assert direction[13] == -Q(1, 4)
    assert all(direction[i] == 0 for i in range(9, 18) if i != 13)
    selected = tuple(Q(edge in cycles[1]) for edge in edges)
    # With both phase directions equal, the factor p*q*(p+q)/2 is p^3.
    # Divide by two to isolate the coefficient of a*t*b*(t-s)*(a*t+b*(t-s)).
    cubic = tuple(
        value / 2
        for value in _edge_cubic(edges, degrees, direction, direction, selected)
    )
    assert cubic[22] == 0  # A direct receiver cubic term is class blind.
    assert cubic[13] == -Q(1, 256)
    assert cubic[12] == cubic[14] == Q(1, 256)
    assert {i for i, value in enumerate(cubic) if value} == {12, 13, 14}
    propagated = tuple(-value for value in _mv(normalized, cubic))
    assert propagated[22] == -Q(1, 768)


@pytest.mark.parametrize(
    "a,b,s,u",
    [(Q(2, 7), Q(3, 11), Q(2, 5), Q(7, 6)), (Q(-1, 3), Q(2, 5), Q(0), Q(4, 7))],
)
def test_expanded_heat_onset_agrees_with_exact_quartic_quadrature(a, b, s, u):
    def integrand(v):
        return (u - v) * (s + v) * v * (a * (s + v) + b * v)

    # Boole's rule is exact for this degree-four integrand, with rational nodes.
    integral = (
        u
        / 90
        * sum(
            weight * integrand(u * i / 4) for i, weight in enumerate((7, 32, 12, 32, 7))
        )
    )
    expanded = a * s**2 * u**3 / 6 + (2 * a + b) * s * u**4 / 12 + (a + b) * u**5 / 20
    assert integral == expanded
    factor = -a * b * integral / 768
    scale = Q(3, 7)
    scaled = (
        -a
        * b
        / 768
        * (
            a * (scale * s) ** 2 * (scale * u) ** 3 / 6
            + (2 * a + b) * (scale * s) * (scale * u) ** 4 / 12
            + (a + b) * (scale * u) ** 5 / 20
        )
    )
    assert scaled == scale**5 * factor


def test_cross_class_noise_and_separate_alternative_have_distinct_budgets():
    delta = Q(1, 37)
    mixed = (1, -1, -1, 1)
    contrast = mixed + tuple(-value for value in mixed)
    errors = tuple(
        delta * sum(c * e for c, e in zip(contrast, corner))
        for corner in product((-1, 1), repeat=8)
    )
    assert min(errors) == -8 * delta and max(errors) == 8 * delta
    # At the strict threshold two class-specific four-record sets can meet:
    # their complete vectors differ by 2delta in each signed mixed direction.
    first = tuple(delta * c for c in mixed)
    second = tuple(-value for value in first)
    assert sum(c * (x - y) for c, x, y in zip(mixed, first, second)) == 8 * delta
    common = (Q(0),) * 4
    assert all(abs(x - y) <= delta for x, y in zip(first, common))
    assert all(abs(x - y) <= delta for x, y in zip(second, common))
    # A separate eight-reading class-blind alternative has its own allowance.
    assert 9 * delta - 8 * delta > 0
    assert 9 * delta - 8 * delta <= 8 * delta
    assert 16 * delta - 8 * delta == 8 * delta
    assert 17 * delta - 8 * delta > 8 * delta


def test_matched_preparation_does_not_impose_equal_source_cost(geometry):
    cycles, _, _, _ = geometry

    def cost(mediator_class):
        x = tuple(Q(k * (j - 4)) for k in (1, mediator_class, 1) for j in range(9))
        assert all(sum(x[9 * c : 9 * c + 9]) == 0 for c in range(3))
        return sum((x[i] - x[j]) ** 2 / 2 for i, j in sum(cycles, ()))

    # This is the common prepared ramp with its common scale removed.
    assert cost(2) == 2 * cost(1)
    assert cost(1) > 0


def test_selected_edge_tail_constant_needs_no_mediator_edge_count(geometry, report):
    cycles, edges, degrees, _ = geometry
    a, b, eta = Q(2, 7), Q(3, 11), Q(1, 13)
    first = tuple(a if i == 11 else -a for i in range(27))
    second = tuple(b if i == 11 else -b for i in range(27))
    selected = tuple(Q(edge in cycles[1]) for edge in edges)
    exact = _edge_cubic(edges, degrees, first, second, selected)
    approximate = _edge_cubic(
        edges,
        degrees,
        tuple((1 + eta) * value for value in first),
        tuple((1 + eta) * value for value in second),
        selected,
    )
    norm = 4 * (a**2 * b + a * b**2)
    assert max(map(abs, exact)) == norm
    # A unit Markov row at node11 and a row perturbation eta at that node
    # saturate the four-factor envelope (three phases plus heat propagation).
    assert (1 + eta) * approximate[11] - exact[11] == norm * ((1 + eta) ** 4 - 1)
    fixed_eta = (2 * report.total_duration) ** 33 / factorial(33)
    fixed_a, fixed_b = report.first_probe_amplitude, report.second_probe_amplitude
    expected = (
        4
        * (abs(fixed_a) ** 2 * abs(fixed_b) + abs(fixed_a) * abs(fixed_b) ** 2)
        * (report.total_duration - report.delay)
        * ((1 + fixed_eta) ** 4 - 1)
    )
    assert report.heat_uniform_tail_upper_bound == fixed_eta
    assert report.mediator_heat_truncation_error_upper_bound == expected


def test_independent_heat_integral_recovers_class_channel_difference(geometry, report):
    cycles, edges, degrees, normalized = geometry
    roots = np.sqrt(np.array(degrees))
    generator = np.array(normalized, dtype=float)
    symmetric = generator * roots[:, None] / roots[None, :]
    eigenvalues, vectors = np.linalg.eigh(symmetric)

    def heat(time):
        symmetric_heat = (vectors * np.exp(-time * eigenvalues)) @ vectors.T
        return symmetric_heat * roots[None, :] / roots[:, None]

    a, b, s, t = map(
        float,
        (
            report.first_probe_amplitude,
            report.second_probe_amplitude,
            report.delay,
            report.total_duration,
        ),
    )
    donor = np.eye(27)[:, 4]
    c1, c2 = np.cos(2 * np.pi / 9), np.cos(4 * np.pi / 9)
    class1 = tuple(c1 if edge in sum(cycles, ()) else 1 for edge in edges)
    class2 = tuple(
        c2 if edge in cycles[1] else c1 if edge in sum(cycles, ()) else 1
        for edge in edges
    )
    selected = tuple(int(edge in cycles[1]) for edge in edges)
    midpoint, halfwidth = (s + t) / 2, (t - s) / 2
    nodes, weights = np.polynomial.legendre.leggauss(48)
    integrals = np.zeros(3)
    for node, weight in zip(nodes, weights):
        time = midpoint + halfwidth * node
        first = a * (donor - heat(time)[:, 4])
        second = b * (donor - heat(time - s)[:, 4])
        row = heat(t - time)[22, :]
        for index, edge_weights in enumerate((class1, class2, selected)):
            forcing = _edge_cubic(edges, degrees, first, second, edge_weights)
            integrals[index] += halfwidth * weight * np.dot(row, forcing)
    assert np.isclose(
        integrals[0] - integrals[1], (c1 - c2) * integrals[2], rtol=1e-11, atol=0
    )
    assert integrals[2] < 0
    np.testing.assert_allclose(
        integrals[2],
        float(report.mediator_heat_polynomial_coefficient),
        rtol=1e-10,
        atol=1e-30,
    )
    # Independent spectral quadrature checks the coefficient; the rational
    # polynomial and its contraction tail remain the rigorous enclosure.
    direct = (integrals[0] - integrals[1]) / (1023 * np.pi) ** 4
    retained_midpoint = sum(report.gamma_fourth_scaled_heat_contrast_bounds) / 2
    assert abs(direct - float(retained_midpoint)) < abs(direct) * 1e-10


def _independent_remainder(amplitude, time, gamma):
    d, ell = 1 - 2 * gamma**2 * time**2, 1 - 2 * gamma * time
    shifted_cubic = 8 * gamma**6 * amplitude**3 * time**3 / (3 * d**3)
    odd_feedback = 8 * gamma**6 * amplitude**3 * time**3 / (3 * d**3 * ell)
    fifth_sine = Q(4, 15) * gamma**6 * amplitude**5 * time / d**5
    c1 = Q(4, 3) * gamma**4 * amplitude**3 / (d**3 * ell)
    c3 = Q(8, 3) * gamma**6 * amplitude**3 / (d**3 * ell**2)
    tangent_feedback = 4 * gamma**2 * (c1 * time**3 / 6 + c3 * time**5 / 20)
    return shifted_cubic + odd_feedback + fifth_sine + tangent_feedback


def test_declared_design_is_unresolved_not_a_class_independence_certificate(report):
    a, b, t, eps, delta, g = (
        report.first_probe_amplitude,
        report.second_probe_amplitude,
        report.total_duration,
        report.endpoint_radius,
        report.readout_error_bound,
        report.gamma_bounds.hi,
    )
    per_class = sum(
        _independent_remainder(amplitude, t, g)
        for amplitude in (abs(a), abs(b), abs(a) + abs(b))
    )
    source = 8 * eps / (1 - 2 * g * t)
    assert report.per_class_mixed_remainder_upper_bound == per_class
    assert report.ideal_contrast_remainder_upper_bound == 2 * per_class
    assert report.source_contrast_error_upper_bound == source
    lower, upper = report.gamma_fourth_scaled_heat_contrast_bounds
    assert -8 * delta < lower <= upper < 0
    assert 2 * per_class > -lower
    true = lower - 2 * per_class - source, upper + 2 * per_class + source
    assert report.true_contrast_bounds == true
    assert true[0] < -8 * delta < 0 < 8 * delta < true[1]
    assert report.recorded_contrast_bounds == (true[0] - 8 * delta, true[1] + 8 * delta)
    assert report.status == "bounds_only"
    assert not report.exact_contrast_zero
    assert not report.true_contrast_sign_certified
    assert not report.recorded_contrast_sign_certified
    assert not report.zero_contrast_record_sets_disjoint
    assert report.strict_recorded_sign_noise_ceiling is None
    # The outer band leaves BOTH detectable and cancelling values possible;
    # this arithmetic is neither an existence nor an overlap proof.
    assert report.all_identities_certified
    assert report.all_work_within_allowances


def test_two_class_event_guards_remain_valid_when_response_is_unresolved(report):
    a, b, s, t, eps, g = (
        report.first_probe_amplitude,
        report.second_probe_amplitude,
        report.delay,
        report.total_duration,
        report.endpoint_radius,
        Q(1, 3000),
    )
    presecond = (a + eps + 2 * g * s * eps) / (1 - 2 * g**2 * s**2)
    first = Q(3, 2) * a**2 + 6 * a * eps
    second = Q(3, 2) * b**2 + 6 * b * presecond
    assert first < report.first_probe_work_allowance
    assert second < report.second_probe_work_allowance
    assert 22 * eps**2 + first + second < report.radius**2 / 2700
    form = (a + b + eps + 2 * g * t * eps) / (1 - 2 * g**2 * t**2)
    phase = eps + 2 * g * t * form
    assert 27 * (form**2 + phase**2) < report.radius**2
    assert 8 * eps**2 <= report.contact_work_allowance
    assert len(report.uniform_history_bounds) == 4
    assert all(history.identity_certified for history in report.uniform_history_bounds)
    assert all(
        history.work_within_allowances for history in report.uniform_history_bounds
    )


def test_independent_remainder_method_floor_survives_perfect_source_and_readout(report):
    amplitude = abs(report.first_probe_amplitude) + abs(report.second_probe_amplitude)
    time = report.total_duration
    # D<=1 and L<=1 imply each combined-amplitude term alone is at least
    # (8/3+32/9)*gamma^6*A^3*T^3. Both classes contribute such a term.
    floor = 2 * Q(56, 9) * report.gamma_bounds.lo**6 * amplitude**3 * time**3
    assert floor > Q(7, 10**29)
    assert report.ideal_contrast_remainder_upper_bound >= floor
    heat = report.gamma_fourth_scaled_heat_contrast_bounds
    assert max(map(abs, heat)) < Q(702, 10**32)
    # Removing source and readout uncertainty cannot remove this method's
    # already larger ideal remainder. This is not a lower bound on the
    # actual full-law error and proves neither overlap nor class equality.
    perfect_source_lower = heat[0] - report.ideal_contrast_remainder_upper_bound
    perfect_source_upper = heat[1] + report.ideal_contrast_remainder_upper_bound
    assert perfect_source_lower < 0 < perfect_source_upper
