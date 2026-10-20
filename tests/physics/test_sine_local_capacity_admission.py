"""Independent static controls for the local-capacity temporal-admission proof.

The exact unsplit member is declared directly from its mathematical state.
Boundary values check algebra, not an empirical parameter search. Directional
derivatives below differentiate the instantaneous full right-hand side; they
neither integrate a trajectory nor produce sampled evidence or replay a source.
"""

from fractions import Fraction as Q
from itertools import product

import mpmath
import networkx as nx
import pytest

from tnfr.dynamics import relational
from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.operators._thol_config import resolve_thol_bifurcation_threshold
from tnfr.physics.relational_observations import bound_relational_sample_jet_budget
from tnfr.physics.relational_sine_support import bound_relational_sine_support_exchange


def _mp(mp, value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 100
    return context


@pytest.fixture(scope="module", autouse=True)
def no_evolution():
    def forbidden(*args, **kwargs):
        pytest.fail("local-capacity static controls must not advance a trajectory")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(relational, "step_relational_exchange", forbidden)
        patch.setattr(relational, "_advance", forbidden)
        yield


def _nominal(mp, *, mean=Q(5, 16), factor=1):
    m, b = _mp(mp, mean), mp.mpf(1) / 1024
    alpha = 2 * mp.pi / 5
    omega = 1 - b / (4 * (2 * m + b + _mp(mp, Q(1e-9))))
    edges = [
        (offset + 2 * k + left, offset + 2 * ((k + 1) % 5) + right, mp.mpf(1))
        for offset in (0, 10)
        for k in range(5)
        for left, right in product(range(2), repeat=2)
    ]
    edges.append((0, 1, mp.mpf(1)))
    edges.extend((left, right, omega) for left in (0, 1) for right in (10, 11))
    forms = [m + (b if i < 2 else 0) for i in range(20)]
    phases = [(i % 10 // 2) * alpha for i in range(20)]
    for i in (2, 3):
        phases[i] = (1 - b) * alpha
    for i in (8, 9):
        phases[i] = (4 + b) * alpha
    capacities = [mp.mpf(1) / 2 if i < 2 else mp.mpf(1) for i in range(20)]
    capacities[2] = _mp(mp, factor)
    return forms, phases, capacities, edges


def _rhs(mp, forms, phases, capacities, edges):
    gradient, current, degree = [mp.mpf(0)] * 20, [mp.mpf(0)] * 20, [0] * 20
    for left, right, weight in edges:
        flux = weight * (forms[left] - forms[right])
        sine = mp.sin(phases[right] - phases[left])
        gradient[left] += flux
        gradient[right] -= flux
        current[left] += sine
        current[right] -= sine
        degree[left] += 1
        degree[right] += 1
    pressure = [current[i] / (mp.pi * degree[i]) for i in range(20)]
    return (
        [capacities[i] * pressure[i] for i in range(20)],
        [capacities[i] * gradient[i] / (mp.pi * degree[i]) for i in range(20)],
        pressure,
    )


def _directional_acceleration(mp, state, target=2):
    forms, phases, capacities, edges = state
    form_rate, phase_rate, _ = _rhs(mp, *state)
    return mp.diff(
        lambda s: _rhs(
            mp,
            [x + s * v for x, v in zip(forms, form_rate)],
            [theta + s * v for theta, v in zip(phases, phase_rate)],
            capacities,
            edges,
        )[0][target],
        0,
    )


@pytest.mark.parametrize("mean", [Q(9, 32), Q(5, 16), Q(11, 32)])
@pytest.mark.parametrize("factor", [Q(128), Q(256)])
def test_nominal_local_acceleration_matches_independent_full_rhs_derivative(
    mp, mean, factor
):
    state = _nominal(mp, mean=mean, factor=factor)
    actual = _directional_acceleration(mp, state)
    alpha, b, c = 2 * mp.pi / 5, mp.mpf(1) / 1024, _mp(mp, factor)
    k, ell = mp.cos((1 - b) * alpha), mp.cos((1 + b) * alpha)
    omega = state[3][-1][2]
    expected = c * b / (2 * mp.pi**2) * (k * (2 + omega) / 7 + c * (k + ell) / 2)
    assert abs(actual - expected) < mp.mpf("1e-94")
    assert actual > c**2 / 2**17 > _mp(mp, Q(resolve_thol_bifurcation_threshold({})))
    pressure = _rhs(mp, *state)[2][2]
    expected_pressure = mp.cos(alpha) * mp.sin(b * alpha) / mp.pi
    assert abs(pressure - expected_pressure) < mp.mpf("1e-95")
    assert pressure > mp.mpf(1) / 20000


def test_local_capacity_is_not_the_uniform_clock_comparison(mp):
    original = _nominal(mp)
    local = _nominal(mp, factor=128)
    uniform = (*original[:2], [128 * nu for nu in original[2]], original[3])
    base_rates = _rhs(mp, *original)
    local_rates = _rhs(mp, *local)
    for row in range(2):
        for i in range(20):
            expected = base_rates[row][i] * (128 if i == 2 else 1)
            assert abs(local_rates[row][i] - expected) < mp.mpf("1e-95")
    # A nonzero untouched phase row forces common clock factor 1; the target
    # phase row forces 128. The discrepancy is also present in acceleration.
    assert base_rates[1][0] != 0 and base_rates[1][2] != 0
    base_acceleration = _directional_acceleration(mp, original)
    uniform_acceleration = _directional_acceleration(mp, uniform)
    local_acceleration = _directional_acceleration(mp, local)
    assert abs(uniform_acceleration - 128**2 * base_acceleration) < mp.mpf("1e-92")
    assert local_acceleration < uniform_acceleration
    alpha, b = 2 * mp.pi / 5, mp.mpf(1) / 1024
    expected_gap = (
        128
        * 127
        * b
        * mp.cos((1 - b) * alpha)
        * (2 + local[3][-1][2])
        / (14 * mp.pi**2)
    )
    assert abs(uniform_acceleration - local_acceleration - expected_gap) < mp.mpf(
        "1e-93"
    )


@pytest.fixture(scope="module")
def represented_reports(mp):
    forms, phases, capacities, edges = _nominal(mp)
    graph = nx.Graph()
    for i in range(20):
        graph.add_node(
            i, EPI=float(forms[i]), theta=float(phases[i]), nu_f=float(capacities[i])
        )
    graph.add_weighted_edges_from((i, j, float(w)) for i, j, w in edges)
    model = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
    before = bound_relational_sine_support_exchange(graph, reference_model=model)
    graph.nodes[2]["nu_f"] = 128
    after = bound_relational_sine_support_exchange(graph, reference_model=model)
    return before, after


def test_shared_report_encloses_derivative_at_its_own_represented_state(
    mp, represented_reports
):
    _, after = represented_reports
    state = (
        [_mp(mp, x) for x in after.epi],
        [_mp(mp, theta) for theta in after.phase],
        [_mp(mp, nu) for nu in after.capacity],
        [
            (i, j, _mp(mp, weight))
            for (i, j), weight in zip(after.edges, after.conductances)
        ],
    )
    oracle = _directional_acceleration(mp, state)
    bounds = after.form_acceleration_bounds()[2]
    assert _mp(mp, bounds.lo) <= oracle <= _mp(mp, bounds.hi)
    assert bounds.lo > Q(resolve_thol_bifurcation_threshold({}))
    pressure = _rhs(mp, *state)[2][2]
    assert _mp(mp, after.pressure[2].lo) <= pressure <= _mp(mp, after.pressure[2].hi)


def test_local_capacity_keeps_storage_but_changes_both_mean_leaves(represented_reports):
    before, after = represented_reports
    for field in (
        "epi",
        "phase",
        "edges",
        "conductances",
        "form_gradient",
        "form_storage",
        "phase_storage",
        "storage",
    ):
        assert getattr(before, field) == getattr(after, field)
    old_weights = tuple(Q(d) / nu for d, nu in zip(before.degrees, before.capacity))
    new_weights = tuple(Q(d) / nu for d, nu in zip(after.degrees, after.capacity))
    assert sum(old_weights) == 104
    assert sum(new_weights) == 100 + Q(4, 128)
    assert [
        i for i, pair in enumerate(zip(old_weights, new_weights)) if pair[0] != pair[1]
    ] == [2]
    for coordinates in (after.epi, after.phase):
        old_mean = sum(w * x for w, x in zip(old_weights, coordinates)) / sum(
            old_weights
        )
        new_mean = sum(w * x for w, x in zip(new_weights, coordinates)) / sum(
            new_weights
        )
        expected_change = (
            (Q(4, 128) - 4) * (coordinates[2] - old_mean) / sum(new_weights)
        )
        assert new_mean - old_mean == expected_change != 0


@pytest.mark.parametrize("factor", [Q(128), Q(256)])
def test_whole_window_bounds_and_shared_budget_jointly_retain_strict_margins(factor):
    epsilon, b, radius = Q(1, 2**18), Q(1, 1024), Q(1, 16)
    # Recompute the proof's coefficient chain. This is a rational bound control,
    # not a statement that sampled endpoints were computed to this precision.
    nominal_acceleration_lower = b * factor**2 * Q(3, 10) * Q(15, 16) / 20
    source_perturbation_upper = factor * (factor + 1) * epsilon / 2
    velocity_coefficient = Q(3, 2) * radius / 3
    phase_acceleration_coefficient = 2 * velocity_coefficient / 3
    jerk_upper = (
        factor
        / 3
        * (
            ((factor + 1) * velocity_coefficient) ** 2
            + factor * (factor + 1) * phase_acceleration_coefficient
        )
    )
    horizon = Q(1, 2**11) / factor
    assert jerk_upper < factor**3 / 128
    acceleration_lower = (
        nominal_acceleration_lower - source_perturbation_upper - jerk_upper * horizon
    )
    assert acceleration_lower > 2 * epsilon * factor**2
    pressure_lower = Q(3, 10) * Q(3, 4) * b * Q(2, 5) - 4 * epsilon / 3
    assert pressure_lower > Q(1, 2**14) > Q(1e-6)
    # Positive acceleration and held capacity imply increasing pressure on
    # this same analytically bounded interval, with no extrapolation beyond it.
    assert acceleration_lower / factor > 0
    step = Q(1, 2**20)
    assert 2 * step <= horizon
    budget = bound_relational_sample_jet_budget(
        sample_step=step,
        sample_error_bounds=(Q(1, 2**52),) * 3,
        third_derivative_bound=jerk_upper,
        observation_time=7,
    )
    assert tuple(Q(float(t)) for t in budget.sample_times) == budget.sample_times
    # Accumulated binary64 subtraction allowances from the declared form
    # range; multiplication by dyadic inverse spacing is exact here.
    slope_error = Q(1, 2**55) / step
    reader_error = (2 * slope_error + Q(1, 2**34)) / step
    assert (
        acceleration_lower - budget.acceleration_sample_error_bound - reader_error
        > Q(resolve_thol_bifurcation_threshold({}))
    )
    assert budget.acceleration_truncation_error_bound > 0
    # This component compares to the instantaneous derivative. The proof's
    # whole-window lower bound instead controls the finite-window average.
    assert budget.acceleration_error_bound > budget.acceleration_sample_error_bound


def test_budget_component_is_the_exact_endpoint_stencil_operator_norm():
    step, epsilon = Q(1, 2**20), Q(1, 2**52)
    budget = bound_relational_sample_jet_budget(
        sample_step=step,
        sample_error_bounds=(epsilon,) * 3,
        third_derivative_bound=Q(256**3, 128),
        observation_time=7,
    )
    extremal_errors = [
        abs(epsilon * (signs[2] - 2 * signs[1] + signs[0]) / step**2)
        for signs in product((-1, 1), repeat=3)
    ]
    assert max(extremal_errors) == budget.acceleration_sample_error_bound
    prior_budget = bound_relational_sample_jet_budget(
        sample_step=step,
        sample_error_bounds=(Q(1, 2**40),) * 3,
        third_derivative_bound=budget.third_derivative_bound,
        observation_time=7,
    )
    assert prior_budget.acceleration_sample_error_bound == 2**12 * max(extremal_errors)
    assert prior_budget.acceleration_sample_error_bound > Q(1, 8)


def test_new_weighted_mean_retains_all_node_form_gate(represented_reports):
    _, source = represented_reports
    weights = tuple(Q(d) / nu for d, nu in zip(source.degrees, source.capacity))
    total = sum(weights)
    normalized = tuple(w / total for w in weights)
    radius, epsilon, b = Q(1, 16), Q(1, 2**18), Q(1, 1024)
    squared_mean_norm = sum(w**2 for w in normalized)
    coefficient = Q(21, 20)
    for weight in normalized:
        assert 1 - 2 * weight + squared_mean_norm < coefficient**2
    # Initial convex-mean bounds and Cauchy-Schwarz propagate the declared
    # identity ball into live EPI margins; no time endpoints are fabricated.
    lower = Q(9, 32) - epsilon - coefficient * radius
    upper = Q(11, 32) + b + epsilon + coefficient * radius
    assert Q(1, 5) < lower < upper < Q(1, 2)
