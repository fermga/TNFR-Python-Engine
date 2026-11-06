"""Independent analytical controls for capacity and temporal access.

Polynomial histories and direct nominal field declarations are mathematical
controls. They neither replay the maintained-source producer nor supply real
observations, an executed capacity event or a THOL birth.
"""

from fractions import Fraction as Q
from itertools import product

import mpmath
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.errors import TNFRValueError
from tnfr.mathematics._rational_interval import I
from tnfr.operators._thol_config import resolve_thol_bifurcation_threshold
from tnfr.operators.nodal_equation import observe_structural_acceleration
from tnfr.physics.relational_observations import bound_relational_sample_jet_budget
from tnfr.physics.relational_sine_support import bound_relational_sine_support_exchange


def _difference(times, values):
    a, b, c = times
    x, y, z = values
    return 2 * ((z - y) / (c - b) - (y - x) / (b - a)) / (c - a)


def _observe(times, values):
    graph = nx.Graph()
    readings = tuple((float(t), float(x)) for t, x in zip(times, values))
    graph.add_node("parent", EPI=readings[-1][1], epi_time_history=readings)
    return observe_structural_acceleration(graph, "parent")


@pytest.mark.parametrize("capacity_factor", [Q(128), Q(256)])
@pytest.mark.parametrize("middle", [Q(1, 4096), Q(1, 8192)])
def test_homologous_genuine_polynomial_windows_scale_quadratically(
    capacity_factor, middle
):
    elapsed = (Q(0), middle, Q(1, 2048))
    times = tuple(7 + t for t in elapsed)
    compressed = tuple(7 + t / capacity_factor for t in elapsed)
    # Smooth mathematical signal; values on homologous windows are identical.
    values = tuple(Q(5, 16) - t**2 / 2**17 for t in elapsed)
    base = _observe(times, values)
    changed = _observe(compressed, values)
    assert Q(base.value) == -Q(1, 2**16)
    assert Q(changed.value) == capacity_factor**2 * Q(base.value)
    assert abs(base.value) < resolve_thol_bifurcation_threshold({}) < abs(changed.value)
    assert tuple(x for _, x in base.samples) == tuple(x for _, x in changed.samples)


def test_equal_clock_windows_are_not_automatically_homologous():
    times = (Q(7), Q(29, 4), Q(15, 2))
    base = tuple(Q(5, 16) + (t - 7) ** 3 / 512 for t in times)
    faster = tuple(Q(5, 16) + (2 * (t - 7)) ** 3 / 512 for t in times)
    original = _difference(times, base)
    changed = _difference(times, faster)
    assert changed == 8 * original
    assert changed != 4 * original


@pytest.mark.parametrize("capacity_factor", [Q(128), Q(192), Q(256)])
def test_existing_endpoint_budget_scales_with_the_signal_and_cannot_certify_access(
    capacity_factor,
):
    budget = bound_relational_sample_jet_budget(
        sample_step=Q(1, 4096) / capacity_factor,
        sample_error_bounds=(Q(1, 2**40),) * 3,
        third_derivative_bound=capacity_factor**3 / 128,
        observation_time=7,
    )
    signal_floor = capacity_factor**2 / 2**17
    # Consume the existing sample-error component. The whole-window theorem
    # already bounds acceleration variation; adding a jet truncation term
    # would compare a different observation with the initial derivative.
    assert budget.acceleration_sample_error_bound == capacity_factor**2 / 2**14
    assert budget.acceleration_sample_error_bound == 8 * signal_floor
    assert budget.acceleration_truncation_error_bound > 0


def test_shorter_common_window_and_separate_hypothetical_accuracy_budget():
    tau = Q(resolve_thol_bifurcation_threshold({}))
    step = Q(1, 2**20)
    epsilon = Q(1, 2**52)
    times = (Q(7), Q(7) + step, Q(7) + 2 * step)
    for capacity_factor in (Q(128), Q(192), Q(256)):
        assert capacity_factor * (times[-1] - 7) <= Q(1, 2048)
        budget = bound_relational_sample_jet_budget(
            sample_step=step,
            sample_error_bounds=(epsilon,) * 3,
            third_derivative_bound=capacity_factor**3 / 128,
            observation_time=7,
        )
        assert budget.acceleration_sample_error_bound == Q(1, 1024)
        assert Q(1, 8) - budget.acceleration_sample_error_bound - Q(1, 8192) > tau
    # Extremal endpoint errors attain the reused component bound; no actual
    # source record with this accuracy is claimed or generated.
    assert max(
        abs(_difference(times, tuple(sign * epsilon for sign in signs)))
        for signs in product((-1, 1), repeat=3)
    ) == Q(1, 1024)


def test_excessive_compression_can_destroy_represented_timestamp_order():
    times = (Q(7), Q(7) + Q(1, 2**60), Q(7) + Q(1, 2**59))
    assert times[0] < times[1] < times[2]
    assert len(set(map(float, times))) == 1
    with pytest.raises(TNFRValueError, match="increase strictly"):
        _observe(times, (Q(5, 16),) * 3)


@pytest.mark.parametrize("mean", [Q(9, 32), Q(5, 16), Q(11, 32)])
def test_exact_nominal_parent_acceleration_has_a_signed_margin(mean):
    mp = mpmath.mp.clone()
    mp.dps = 100
    b = mp.mpf(1) / 1024
    m = mp.mpf(mean.numerator) / mean.denominator
    epsilon = Q(1e-9)
    regularizer = mp.mpf(epsilon.numerator) / epsilon.denominator
    omega = 1 - b / (4 * (2 * m + b + regularizer))
    angle = (1 - b) * 2 * mp.pi / 5
    q_parent = b * (4 + 2 * omega)
    parent_phase_rate = q_parent / (14 * mp.pi)
    old_neighbor_phase_rate = -b / (2 * mp.pi)
    cross_neighbor_phase_rate = -omega * b / (3 * mp.pi)
    acceleration = (
        4 * mp.cos(angle) * (old_neighbor_phase_rate - parent_phase_rate)
        + 2 * (cross_neighbor_phase_rate - parent_phase_rate)
    ) / (14 * mp.pi)
    assert mp.mpf(3) / 4 < omega < 1
    assert mp.mpf(3) / 10 < mp.cos(angle) < mp.mpf(1) / 3
    assert -mp.mpf(1) / 2**15 < acceleration < -mp.mpf(1) / 2**16


def test_whole_window_rational_margins_precede_capacity_comparison():
    # Independent rational controls for inequalities in the analytical proof.
    epsilon = Q(1, 2**18)
    horizon = Q(1, 2**11)
    assert Q(33, 1960 * 1024) > Q(1, 2**16)
    assert Q(35, 4608) < Q(1, 128)
    source_margin = Q(1, 2**16) - epsilon / 3
    assert source_margin - horizon / 128 > Q(1, 2**17)
    assert 128**2 / Q(2**17) == Q(1, 8)
    assert Q(1, 8) > Q(resolve_thol_bifurcation_threshold({}))


def test_equilibrium_is_a_counterexample_to_universal_capacity_triggering():
    graph = nx.path_graph(2)
    for node in graph:
        graph.nodes[node].update(EPI=Q(5, 16), theta=0, nu_f=128)
    report = bound_relational_sine_support_exchange(
        graph,
        reference_model=RelationalExchangeModel(
            1, epi_weight=0, phase_domain="regular"
        ),
    )
    assert report.form_rates == (I(0), I(0))
    assert report.phase_rates == (I(0), I(0))
    assert all(value.contains(Q(0)) for value in report.form_acceleration_bounds())
    # A theoretical field does not populate a timestamped history.
    assert not observe_structural_acceleration(graph, 0).available
