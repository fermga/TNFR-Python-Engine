"""Analytical history controls for the unchanged maintained-contact source.

These manufactured polynomial readings test temporal and numerical claims;
they are not trajectories or sampled evidence from a sine organization. The
source-family proof is SINE_ORGANIZED_BIRTH_OBSTRUCTION. No producer or THOL
event is run here.
"""

from fractions import Fraction as Q
from itertools import product

import networkx as nx
import pytest

from tnfr.mathematics._rational_interval import I, pi_interval, sqrt
from tnfr.operators._thol_config import resolve_thol_bifurcation_threshold
from tnfr.operators.nodal_equation import observe_structural_acceleration

_TIMES = (Q(7), Q(29, 4), Q(15, 2))


def _difference(times, values):
    a, b, c = times
    x, y, z = values
    return 2 * ((z - y) / (c - b) - (y - x) / (b - a)) / (c - a)


def _integral_power(exponent, start, stop):
    return (stop ** (exponent + 1) - start ** (exponent + 1)) / (exponent + 1)


def _kernel_moment(exponent, times):
    """Integrate s**exponent against the two explicit triangular pieces."""
    a, b, c = times
    left = _integral_power(exponent + 1, a, b) - a * _integral_power(exponent, a, b)
    right = c * _integral_power(exponent, b, c) - _integral_power(exponent + 1, b, c)
    return 2 * (left / (b - a) + right / (c - b)) / (c - a)


def _observe(times, values):
    graph = nx.Graph()
    readings = tuple((float(t), float(x)) for t, x in zip(times, values))
    graph.add_node("parent", EPI=readings[-1][1], epi_time_history=readings)
    return observe_structural_acceleration(graph, "parent")


@pytest.mark.parametrize("times", [_TIMES, (Q(0), Q(1), Q(3)), (Q(2), Q(5, 2), Q(9))])
@pytest.mark.parametrize("degree", [2, 3, 4, 5])
def test_nonuniform_kernel_has_unit_mass_and_matches_independent_polynomial(
    times, degree
):
    assert _kernel_moment(0, times) == 1
    values = tuple(t**degree for t in times)
    averaged_derivative = degree * (degree - 1) * _kernel_moment(degree - 2, times)
    assert _difference(times, values) == averaged_derivative
    lower, upper = (
        degree * (degree - 1) * t ** (degree - 2) for t in (times[0], times[-1])
    )
    assert lower <= averaged_derivative <= upper


@pytest.mark.parametrize("degree", [2, 3])
def test_live_reader_matches_exact_dyadic_polynomial_window(degree):
    values = tuple(Q(5, 16) + (t - 7) ** degree / 512 for t in _TIMES)
    observation = _observe(_TIMES, values)
    assert observation.available
    assert Q(observation.value) == _difference(_TIMES, values)
    assert abs(observation.value) < resolve_thol_bifurcation_threshold({})


def test_parent_and_all_node_bounds_have_a_strict_default_policy_margin():
    factor = sqrt(I(2)) * Q(1, 16) / (pi_interval() * pi_interval())
    parent = factor * Q(3, 4)
    all_nodes = factor * 2
    assert parent.hi < Q(1, 128)
    assert all_nodes.hi < Q(1, 48)
    tau = Q(resolve_thol_bifurcation_threshold({}))
    assert Q(1, 48) < Q(1, 10) <= tau
    # A dimensionful default is policy; this is not a universal acceleration cap.
    p2_acceleration = 2 / (pi_interval() * pi_interval())
    assert p2_acceleration.lo > tau


def test_conserved_weighted_mean_and_identity_bound_keep_readings_in_range():
    # A weighted mean is inside the initial coordinate hull; the identity
    # bounds each later coordinate-to-mean difference by sqrt(2)*r.
    excursion = sqrt(I(2)) / 16
    lower = Q(9, 32) - Q(1, 512) - excursion
    upper = Q(11, 32) + Q(1, 1024) + Q(1, 512) + excursion
    epsilon = Q(1, 2**40)
    assert lower.lo - epsilon > Q(1, 8)
    assert upper.hi + epsilon < Q(1, 2)


def test_fixed_window_reading_error_bound_is_attained_at_a_signed_corner():
    epsilon = Q(1, 2**40)
    errors = tuple(
        abs(_difference(_TIMES, tuple(sign * epsilon for sign in signs)))
        for signs in product((-1, 1), repeat=3)
    )
    assert max(errors) == 64 * epsilon
    allowance = Q(1, 128) + 64 * epsilon + Q(1, 2**48)
    assert allowance < Q(resolve_thol_bifurcation_threshold({}))
    # The actual binary64 reader is exact on these dyadic constant-signal corners.
    for signs in product((-1, 1), repeat=3):
        values = tuple(Q(5, 16) + sign * epsilon for sign in signs)
        assert Q(_observe(_TIMES, values).value) == _difference(_TIMES, values)


def test_nonuniform_sample_error_weights_are_all_required():
    times = (Q(0), Q(1, 4), Q(1))
    radii = (Q(1, 1024), Q(1, 2048), Q(1, 4096))
    expected = 2 * ((radii[0] + radii[1]) / Q(1, 4) + (radii[1] + radii[2]) / Q(3, 4))
    actual = max(
        abs(
            _difference(
                times, tuple(sign * radius for sign, radius in zip(signs, radii))
            )
        )
        for signs in product((-1, 1), repeat=3)
    )
    assert actual == expected


def test_tiny_windows_can_amplify_rounding_of_a_zero_acceleration_signal():
    h = Q(1, 2**30)
    times = (Q(0), h, 2 * h)
    values = tuple(Q(5, 16) + t / 2**25 for t in times)
    assert _difference(times, values) == 0
    # Round-to-nearest-even collapses the half-ulp middle increment. These are
    # readings of a linear mathematical control, not evidence of TNFR motion.
    observation = _observe(times, values)
    assert observation.value == 64
    assert observation.value > resolve_thol_bifurcation_threshold({})
    errors = tuple(abs(Q(float(x)) - x) for x in values)
    assert max(errors) == Q(1, 2**55)


def test_a_spatial_bound_does_not_supply_missing_temporal_evidence():
    graph = nx.Graph()
    graph.add_node("parent", EPI=Q(5, 16), nu_f=Q(1, 2))
    observation = observe_structural_acceleration(graph, "parent")
    assert not observation.available
    assert observation.value is None
