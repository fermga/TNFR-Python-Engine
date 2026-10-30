"""Static complete-law current identities and equal-observation nonclosure.

The two witness states are constructed points in a larger identity-shaped
class. These controls neither replay a preparation nor integrate any flow.
"""

import os
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
from itertools import repeat

import mpmath
import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import I
from tnfr.mathematics._rational_interval import cos as interval_cos
from tnfr.mathematics._rational_interval import pi_interval
from tnfr.operators import _coupling_stage_kernel, network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _mixed_cycle_current as owner
from tnfr.physics import _mixed_phase_sensitive_interface as phase_owner
from tnfr.physics import _sine_flow as flow_owner
from tnfr.research.artifact_io import encode_exact_tree, write_json_once

DELTA, WEIGHT_ERROR = Q(1, 1024), Q(2, 10**60)
AMPLITUDE = DELTA / 10000
REGIONS = tuple(tuple(range(k, k + 9)) for k in (0, 9, 18, 27))
CYCLES = tuple(
    tuple((k + j, k + (j + 1) % 9) for j in range(9)) for k in (0, 9, 18, 27)
)
TOL = Q(1, 10**190)
CLOCK_FACTOR = Q(1023, 1024)


def _mp(mp, value):
    if isinstance(value, (int, Q, np.integer)):
        value = Q(value)
        return mp.mpf(value.numerator) / value.denominator
    return mp.mpf(value)


def _dot(first, second):
    return sum(a * b for a, b in zip(first, second, strict=True))


def _center(values):
    mean = sum(values) / len(values)
    return tuple(value - mean for value in values)


def _turns():
    return tuple(Q((1 + j // 9) * (j % 9 - 4), 9) for j in range(18))


def _support():
    bins = tuple(int((9 * turn) % 9) for turn in _turns())
    old = tuple(
        sorted(
            tuple(sorted((9 * p + j, 9 * p + (j + 1) % 9)))
            for p in (0, 1)
            for j in range(9)
        )
    )
    edges = {
        (a, b)
        for a in range(18)
        for b in range(a + 1, 18)
        if min((bins[b] - bins[a]) % 9, (bins[a] - bins[b]) % 9) in (1, 2)
        or (a // 9 != b // 9 and bins[a] == bins[b])
    }
    return old, tuple(sorted(edges - set(old)))


def _nominal_weights(mp):
    theta = tuple(
        2 * mp.pi * _mp(mp, turn) + (-1 if j < 9 else 1) * _mp(mp, DELTA) / 2
        for j, turn in enumerate(_turns())
    )
    return tuple(
        1 - abs((theta[b] - theta[a] + mp.pi) % (2 * mp.pi) - mp.pi) / (2 * mp.pi)
        for a, b in _support()[1]
    )


def _weights(mp, side):
    values = tuple(
        Q(mp.nstr(value, 210))
        + (1 if value < _mp(mp, Q(7, 8)) else -1)
        * WEIGHT_ERROR
        * Q(1 + (j + side) % 3, 6)
        for j, value in enumerate(_nominal_weights(mp))
    )
    assert all(Q(3, 4) <= value <= 1 for value in values)
    return values


def _edges(weights):
    old, new = _support()
    return tuple((a, b, Q(1)) for a, b in old) + tuple(
        (a, b, weight) for (a, b), weight in zip(new, weights, strict=True)
    )


def _h_apply(mp, values, weights):
    out = [mp.mpf(0)] * 36
    for side in (0, 1):
        for a, b, weight in _edges(weights[side]):
            a, b = a + 18 * side, b + 18 * side
            current = _mp(mp, weight) * (values[a] - values[b])
            out[a] += current
            out[b] -= current
    return tuple(out)


def _sine(mp, phases):
    theta = tuple(
        2 * mp.pi * _mp(mp, turn) + _mp(mp, phase)
        for turn, phase in zip(_turns() * 2, phases, strict=True)
    )
    out = [mp.mpf(0)] * 36
    for side in (0, 1):
        for a, b, _ in _edges((Q(1),) * 63):
            a, b = a + 18 * side, b + 18 * side
            current = mp.sin(theta[b] - theta[a])
            out[a] += current
            out[b] -= current
    return tuple(out)


def _exchange(values, slots=18):
    first = tuple(values[j] - values[18 + j] if j < slots else 0 for j in range(18))
    return first + tuple(-value for value in first)


def _field(mp, forms, phases, weights):
    x = tuple(_mp(mp, value) for value in forms)
    hx, sine, gx = _h_apply(mp, x, weights), _sine(mp, phases), _exchange(x)
    gamma = 1 / (1023 * mp.pi)
    cross = tuple(
        mp.sin(_mp(mp, phases[j + 18]) - _mp(mp, phases[j])) for j in range(18)
    )
    cross = cross + tuple(-value for value in cross)
    lx = tuple(a + 9 * b for a, b in zip(hx, gx, strict=True))
    splus = tuple(a + 9 * b for a, b in zip(sine, cross, strict=True))
    return tuple((-a + gamma * b) / 9 for a, b in zip(lx, splus, strict=True)), tuple(
        gamma * a / 9 for a in lx
    )


def _state(forms, phases):
    return tuple(I(value) for value in forms) + tuple(
        2 * pi_interval() * turn + I(value)
        for turn, value in zip(_turns() * 2, phases, strict=True)
    )


def _new_field(weights):
    return phase_owner._mixed_phase_sensitive_field(
        first_new_edge_conductances=weights[0],
        second_new_edge_conductances=weights[1],
    )


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_event_or_retained_producer():
    static_factory = flow_owner._sine_rate_evaluator

    def forbidden(*args, **kwargs):
        pytest.fail("cycle-current controls cannot replay or advance a source")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for module, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (Operator, "__call__"),
            (_coupling_stage_kernel, "propose_coupling_stage"),
            (_coupling_stage_kernel, "propose_coupling_target"),
            (network_stage, "execute_coupling_stage"),
            (network_stage, "execute_neighbor_stage"),
            (network_stage, "execute_pointwise_stage"),
            (network_stage, "execute_operator_major_stage"),
            (network_stage, "_commit_coupling_structure"),
            (word_execution, "execute_network_operator_stage"),
        ):
            patch.setattr(module, name, forbidden)
        # Permit the existing row builder for static full-state comparisons;
        # every integrator, acquisition producer and operator remains blocked.
        with pytest.MonkeyPatch.context() as static:
            static.setattr(flow_owner, "_sine_rate_evaluator", static_factory)
            yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 220

    def forbidden(*args, **kwargs):
        pytest.fail("cycle-current proof cannot evaluate an exponential or root")

    with pytest.MonkeyPatch.context() as patch:
        for name in ("exp", "expm", "eig", "polyroots", "odefun", "invertlaplace"):
            patch.setattr(context, name, forbidden)
        yield context


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)


def _complete_edges(weights):
    return tuple(
        sorted(
            tuple(
                (a + offset, b + offset, weight, Q(1))
                for offset, row in zip((0, 18), weights, strict=True)
                for a, b, weight in _edges(row)
            )
            + tuple((j, j + 18, Q(9), Q(9)) for j in range(18))
        )
    )


def _divergence(edges, currents):
    values = [0] * 36
    for (a, b, *_), current in zip(edges, currents, strict=True):
        values[a] += current
        values[b] -= current
    return tuple(values)


def _cycle_sum(edges, currents, cycle):
    indexed = {
        (a, b): current for (a, b, *_), current in zip(edges, currents, strict=True)
    }
    return sum(indexed[(a, b)] if a < b else -indexed[(b, a)] for a, b in cycle) / 9


def _oracle(mp, forms, offsets, weights):
    x = tuple(_mp(mp, value) for value in forms)
    theta = tuple(
        2 * mp.pi * _mp(mp, turn) + _mp(mp, offset)
        for turn, offset in zip(_turns() * 2, offsets, strict=True)
    )
    dx, dy = _field(mp, forms, offsets, weights)
    edges = _complete_edges(weights)
    gamma = 1 / (1023 * mp.pi)
    currents = tuple(
        (
            _mp(mp, w) * (x[a] - x[b])
            + gamma * _mp(mp, sigma) * mp.sin(theta[a] - theta[b])
        )
        / 9
        for a, b, w, sigma in edges
    )
    phase_currents = tuple(
        -gamma * _mp(mp, w) * (x[a] - x[b]) / 9 for a, b, w, _ in edges
    )
    derivatives = tuple(
        (
            _mp(mp, w) * (dx[a] - dx[b])
            + gamma * _mp(mp, sigma) * mp.cos(theta[a] - theta[b]) * (dy[a] - dy[b])
        )
        / 9
        for a, b, w, sigma in edges
    )
    return dict(
        x=x,
        theta=theta,
        dx=dx,
        dy=dy,
        edges=edges,
        currents=currents,
        phase_currents=phase_currents,
        derivatives=derivatives,
        gamma=gamma,
    )


def _observe(forms, offsets, weights):
    return owner._observe_mixed_cycle_current(
        forms=forms,
        phase_lifts=_state((Q(0),) * 36, offsets)[36:],
        first_new_edge_conductances=weights[0],
        second_new_edge_conductances=weights[1],
    )


def _interval_record(interval):
    return {"lo": interval.lo, "hi": interval.hi}


@pytest.fixture(scope="module")
def points(mp):
    weights = (_weights(mp, 0), _weights(mp, 1))
    forms = _center(tuple(Q((7 * j) % 13 - 6, 10**5) for j in range(36)))
    offsets = tuple(Q((11 * j) % 17 - 8, 10**5) + Q(j // 9 - 1, 100) for j in range(36))
    generic = _observe(forms, offsets, weights)
    nominal = _observe((Q(0),) * 36, (Q(0),) * 36, weights)
    hidden = (AMPLITUDE, -AMPLITUDE) + (Q(0),) * 34
    omega = 2 * pi_interval() / 9
    d = interval_cos(omega - 2 * AMPLITUDE) - interval_cos(omega + AMPLITUDE)
    k = interval_cos(omega) - interval_cos(omega + AMPLITUDE)
    gradient = (d, -d, k) + (I(0),) * 5 + (-k,) + (I(0),) * 27
    # These boxes contain one declared correlated point x=a*g(theta).
    # They do not assert equal K for every independent member of the boxes.
    witness_forms = tuple(AMPLITUDE * value for value in gradient)
    first = _observe((Q(0),) * 36, hidden, weights)
    second = _observe(witness_forms, hidden, weights)
    result = dict(
        weights=weights,
        forms=forms,
        offsets=offsets,
        generic=generic,
        nominal=nominal,
        hidden=hidden,
        gradient=gradient,
        witness_forms=witness_forms,
        first=first,
        second=second,
    )
    destination = os.environ.get("TNFR_CYCLE_CURRENT_RECORD_PATH")
    if destination:
        record = {
            "schema": "tnfr.cycle-current-control-observations.v1",
            "constructed_static_points": True,
            "acquired_source_state_evaluated": False,
            "new_trajectories": 0,
            "new_measurements": 0,
            "amplitude": AMPLITUDE,
            "witness_scaled_norm_squared_upper_bound": 8 * AMPLITUDE**2
            + 32 * AMPLITUDE**4,
            "witness_marked_perturbation_upper_bound": AMPLITUDE,
            "nominal_cycle_currents": tuple(
                _interval_record(v) for v in nominal.cycle_currents
            ),
            "nominal_cycle_derivatives": tuple(
                _interval_record(v) for v in nominal.cycle_current_derivatives
            ),
            "witness_cycle_currents_first": tuple(
                _interval_record(v) for v in first.cycle_currents
            ),
            "witness_cycle_currents_second": tuple(
                _interval_record(v) for v in second.cycle_currents
            ),
            "witness_cycle_derivatives_first": tuple(
                _interval_record(v) for v in first.cycle_current_derivatives
            ),
            "witness_cycle_derivatives_second": tuple(
                _interval_record(v) for v in second.cycle_current_derivatives
            ),
            "witness_da_derivative_separation": _interval_record(
                second.cycle_current_derivatives[0] - first.cycle_current_derivatives[0]
            ),
            "witness_matching_lower_bound": _interval_record(
                2 * AMPLITUDE * second.gamma_bounds**2 * d**2 / 81
            ),
            "current_units": generic.current_units,
            "cycle_derivative_units": generic.cycle_derivative_units,
            "empirical_validation": False,
        }
        write_json_once(destination, encode_exact_tree(record))
    return result


def test_complete_currents_and_derivatives_match_independent_180_edges_and_72_rows(
    mp, points
):
    actual = points["generic"]
    expected = _oracle(mp, points["forms"], points["offsets"], points["weights"])
    assert len(actual.edge_pairs) == len(set(actual.edge_pairs)) == 180
    assert actual.edge_pairs == tuple((a, b) for a, b, _, _ in expected["edges"])
    assert actual.form_conductances == tuple(w for _, _, w, _ in expected["edges"])
    assert actual.phase_coefficients == tuple(s for _, _, _, s in expected["edges"])
    for intervals, values in (
        (actual.form_rates + actual.phase_rates, expected["dx"] + expected["dy"]),
        (actual.edge_currents, expected["currents"]),
        (actual.edge_phase_currents, expected["phase_currents"]),
        (actual.edge_current_derivatives, expected["derivatives"]),
    ):
        for interval, value in zip(intervals, values, strict=True):
            _contains(mp, interval, value)
    assert points["weights"][0] != points["weights"][1]
    assert all(s == 1 for a, b, _, s in expected["edges"] if a // 18 == b // 18)
    assert all(w == s == 9 for a, b, w, s in expected["edges"] if a < 18 <= b)


def test_local_regional_and_global_continuity_include_both_inventories(mp, points):
    actual = points["generic"]
    expected = _oracle(mp, points["forms"], points["offsets"], points["weights"])
    for currents, rates, divergence, region_rates in (
        (
            expected["currents"],
            expected["dx"],
            actual.current_divergence,
            actual.region_mean_form_rates,
        ),
        (
            expected["phase_currents"],
            expected["dy"],
            actual.phase_current_divergence,
            actual.region_mean_phase_rates,
        ),
    ):
        div = _divergence(expected["edges"], currents)
        for interval, current, rate in zip(divergence, div, rates, strict=True):
            _contains(mp, interval, current)
            assert abs(current + rate) < _mp(mp, TOL)
        assert abs(sum(rates)) < _mp(mp, TOL)
        for region, reported in zip(REGIONS, region_rates, strict=True):
            selected = set(region)
            outward = sum(
                current * (int(a in selected) - int(b in selected))
                for (a, b, *_), current in zip(expected["edges"], currents, strict=True)
            )
            rate = sum(rates[j] for j in region) / 9
            assert abs(rate + outward / 9) < _mp(mp, TOL)
            _contains(mp, reported, rate)
    _contains(mp, sum(actual.form_rates, I(0)), mp.mpf(0))
    _contains(mp, sum(actual.phase_rates, I(0)), mp.mpf(0))


def test_orientation_transport_cycle_reversal_and_clock_powers(mp, points):
    actual = points["generic"]
    expected = _oracle(mp, points["forms"], points["offsets"], points["weights"])
    edges, currents = expected["edges"], expected["currents"]
    reversed_edges = tuple(
        (b, a, w, s) if index % 3 == 0 else (a, b, w, s)
        for index, (a, b, w, s) in enumerate(edges)
    )
    reversed_currents = tuple(
        -v if index % 3 == 0 else v for index, v in enumerate(currents)
    )
    assert _divergence(edges, currents) == _divergence(
        reversed_edges, reversed_currents
    )
    assert actual.cycle_edges == CYCLES
    factor = _mp(mp, CLOCK_FACTOR)
    for cycle, current, derivative in zip(
        CYCLES, actual.cycle_currents, actual.cycle_current_derivatives, strict=True
    ):
        value = _cycle_sum(edges, currents, cycle)
        reverse = _cycle_sum(edges, currents, tuple((b, a) for a, b in reversed(cycle)))
        assert abs(value + reverse) < _mp(mp, TOL)
        _contains(mp, current, value)
        # If J_t=c*J_tau, then dJ_t/dt=c^2*dJ_tau/dtau, not c*dJ_tau/dtau.
        physical = _cycle_sum(edges, tuple(factor * v for v in currents), cycle)
        physical_derivative = _cycle_sum(
            edges, tuple(factor**2 * v for v in expected["derivatives"]), cycle
        )
        assert abs(physical - factor * value) < _mp(mp, TOL)
        _contains(mp, CLOCK_FACTOR**2 * derivative, physical_derivative)
    assert actual.clock == "tau=(1023/1024)*t; unchanged complete phase-sensitive rows"


def test_divergence_alone_does_not_select_the_constitutive_cycle_current(mp, points):
    expected = _oracle(mp, points["forms"], points["offsets"], points["weights"])
    addition = {}
    amount = _mp(mp, Q(1, 1000))
    for a, b in CYCLES[0]:
        addition[tuple(sorted((a, b)))] = amount if a < b else -amount
    alternative = tuple(
        value + addition.get((a, b), 0)
        for (a, b, *_), value in zip(
            expected["edges"], expected["currents"], strict=True
        )
    )
    original_div = _divergence(expected["edges"], expected["currents"])
    changed_div = _divergence(expected["edges"], alternative)
    assert max(
        abs(a - b) for a, b in zip(original_div, changed_div, strict=True)
    ) < _mp(mp, TOL)
    difference = _cycle_sum(expected["edges"], alternative, CYCLES[0]) - _cycle_sum(
        expected["edges"], expected["currents"], CYCLES[0]
    )
    assert abs(difference - amount) < _mp(mp, TOL)
    assert points["generic"].current_selected_by_supplied_law
    assert not points["generic"].current_selected_by_continuity_alone


def test_stationary_template_has_divergence_free_nonzero_cycle_currents(mp, points):
    actual = points["nominal"]
    expected = _oracle(mp, (Q(0),) * 36, (Q(0),) * 36, points["weights"])
    assert max(abs(value) for value in expected["dx"] + expected["dy"]) < _mp(mp, TOL)
    for values in (
        actual.form_rates,
        actual.phase_rates,
        actual.current_divergence,
        actual.phase_current_divergence,
        actual.cycle_current_derivatives,
    ):
        for interval in values:
            _contains(mp, interval, mp.mpf(0))
    for winding, interval in zip((1, 2, 1, 2), actual.cycle_currents, strict=True):
        expected_current = -expected["gamma"] * mp.sin(2 * mp.pi * winding / 9) / 9
        assert expected_current < 0 and interval.hi < 0
        _contains(mp, interval, expected_current)
    # The actual storage rate is -||Lplus*x||^2/9, zero at this constructed point.
    hx = _h_apply(mp, (mp.mpf(0),) * 36, points["weights"])
    assert _dot(hx, hx) == 0
    assert not actual.supplied_state_authenticated_as_source


def test_cycle_telescoping_and_directional_derivative_use_the_complete_phase_row(
    mp, points
):
    actual = points["generic"]
    expected = _oracle(mp, points["forms"], points["offsets"], points["weights"])
    x, theta, dx, dy, gamma = (
        expected[name] for name in ("x", "theta", "dx", "dy", "gamma")
    )
    for cycle, reported, derivative in zip(
        CYCLES, actual.cycle_currents, actual.cycle_current_derivatives, strict=True
    ):
        assert abs(sum(x[a] - x[b] for a, b in cycle)) < _mp(mp, TOL)
        phase_only = gamma * sum(mp.sin(theta[a] - theta[b]) for a, b in cycle) / 81
        _contains(mp, reported, phase_only)

        def straight_line_observation(t):
            # A local directional derivative at one point, not an ODE trajectory.
            return sum(
                (
                    x[a]
                    + t * dx[a]
                    - x[b]
                    - t * dx[b]
                    + gamma * mp.sin(theta[a] + t * dy[a] - theta[b] - t * dy[b])
                )
                / 81
                for a, b in cycle
            )

        direction = mp.diff(straight_line_observation, mp.mpf(0))
        _contains(mp, derivative, direction)
        edge_chain = _cycle_sum(expected["edges"], expected["derivatives"], cycle)
        assert abs(direction - edge_chain) < _mp(mp, TOL)
    # The same test on a matching edge independently exercises its coefficient9.
    a, b = 4, 22
    index = actual.edge_pairs.index((a, b))

    def matching(t):
        return (
            x[a]
            + t * dx[a]
            - x[b]
            - t * dx[b]
            + gamma * mp.sin(theta[a] + t * dy[a] - theta[b] - t * dy[b])
        )

    _contains(mp, actual.edge_current_derivatives[index], mp.diff(matching, mp.mpf(0)))


def _witness_exact(mp, points):
    a, omega = _mp(mp, AMPLITUDE), 2 * mp.pi / 9
    d = mp.cos(omega - 2 * a) - mp.cos(omega + a)
    k = mp.cos(omega) - mp.cos(omega + a)
    gradient = (d, -d, k) + (mp.mpf(0),) * 5 + (-k,) + (mp.mpf(0),) * 27
    forms = tuple(a * value for value in gradient)
    expected = _oracle(mp, forms, points["hidden"], points["weights"])
    return a, d, k, gradient, expected


def test_equal_regional_readings_have_distinct_complete_law_cycle_derivatives(
    mp, points
):
    first, second = points["first"], points["second"]
    a, d, k, gradient, expected = _witness_exact(mp, points)
    assert d > 0 and k > 0
    assert abs(d - 2 * mp.sin(2 * mp.pi / 9 - a / 2) * mp.sin(3 * a / 2)) < _mp(mp, TOL)
    assert (
        tuple(sum(gradient[j] for j in region) for region in REGIONS)
        == (mp.mpf(0),) * 4
    )
    assert first.phase_lifts == second.phase_lifts
    assert first.first_new_edge_conductances == second.first_new_edge_conductances
    assert first.second_new_edge_conductances == second.second_new_edge_conductances
    # Exact point correlations, rather than an intersection of intervals, give equal K.
    assert first.cycle_currents == second.cycle_currents
    for observation in (first, second):
        for interval in observation.region_mean_forms + observation.region_mean_phases:
            _contains(mp, interval, mp.mpf(0))
    for interval, value in zip(second.forms, expected["x"], strict=True):
        _contains(mp, interval, value)
    for interval, value in zip(
        second.form_rates + second.phase_rates,
        expected["dx"] + expected["dy"],
        strict=True,
    ):
        _contains(mp, interval, value)
    theta = expected["theta"]
    for j in range(9):
        direct = mp.cos(theta[j] - theta[(j + 1) % 9]) - mp.cos(
            theta[(j - 1) % 9] - theta[j]
        )
        assert abs(direct - gradient[j]) < _mp(mp, TOL)
    hx = _h_apply(mp, gradient, points["weights"])
    gx = _exchange(gradient)
    lx = tuple(u + 9 * v for u, v in zip(hx, gx, strict=True))
    quadratic = _dot(gradient, lx)
    separation = a * expected["gamma"] ** 2 * quadratic / 729
    assert separation > 0
    edge_rate = _cycle_sum(expected["edges"], expected["derivatives"], CYCLES[0])
    assert abs(edge_rate - separation) < _mp(mp, TOL)
    _contains(mp, first.cycle_current_derivatives[0], mp.mpf(0))
    interval_separation = (
        second.cycle_current_derivatives[0] - first.cycle_current_derivatives[0]
    )
    assert interval_separation.lo > 0
    _contains(mp, interval_separation, separation)
    assert not first.supplied_state_authenticated_as_source
    assert not second.collective_derivative_closed_on_readings


def test_witness_identity_bounds_and_positive_weight_argument_are_source_independent(
    mp, points
):
    a, d, k, gradient, expected = _witness_exact(mp, points)
    assert d <= 3 * a and k <= a
    gradient_square = _dot(gradient, gradient)
    assert gradient_square <= 20 * a**2
    centered_phase_square = sum(_mp(mp, value) ** 2 for value in points["hidden"])
    scaled_square = _dot(expected["x"], expected["x"]) + 4 * centered_phase_square
    # The maintained proof uses the looser32 coefficient; the direct d,k bound
    # above independently confirms its safety without changing the contract.
    bound = 8 * AMPLITUDE**2 + 32 * AMPLITUDE**4
    assert scaled_square <= _mp(mp, bound)
    assert bound < 9 * AMPLITUDE**2 < Q(1, 500) ** 2
    assert 2 * AMPLITUDE < DELTA / 4
    # Every edge energy is nonnegative for the entire legal held-weight box.
    weighted_energy = sum(
        _mp(mp, w) * (gradient[i] - gradient[j]) ** 2
        for i, j, w, _ in expected["edges"]
    )
    matching_energy = 9 * gradient_square
    assert weighted_energy >= matching_energy > 0
    lower = 2 * a * expected["gamma"] ** 2 * d**2 / 81
    assert a * expected["gamma"] ** 2 * weighted_energy / 729 >= lower > 0
    assert all(Q(3, 4) <= w <= 1 for row in points["weights"] for w in row)
    # Small source-associated weight perturbations are held identically between points;
    # the positivity argument itself needs no source acquisition or nominal equality.
    for row in points["weights"]:
        nominal = _nominal_weights(mp)
        assert max(
            abs(_mp(mp, w) - n) for w, n in zip(row, nominal, strict=True)
        ) <= _mp(mp, WEIGHT_ERROR)
        assert any(_mp(mp, w) != n for w, n in zip(row, nominal, strict=True))
    theta = expected["theta"]
    for cycle, winding in zip(CYCLES, (1, 2, 1, 2), strict=True):
        perturbation = max(
            abs(points["hidden"][b] - points["hidden"][a]) for a, b in cycle
        )
        assert perturbation <= 2 * AMPLITUDE
        wrapped = tuple(
            (theta[b] - theta[a] + mp.pi) % (2 * mp.pi) - mp.pi for a, b in cycle
        )
        assert max(abs(value) for value in wrapped) < mp.pi / 2
        assert abs(sum(wrapped) / (2 * mp.pi) - winding) < _mp(mp, TOL)
    bins = tuple(int((9 * turn) % 9) for turn in _turns())
    for offset in (0, 18):
        for left in range(9):
            right = next(j for j in range(9, 18) if bins[j] == bins[left])
            assert (
                abs(points["hidden"][offset + right] - points["hidden"][offset + left])
                <= AMPLITUDE
            )
    assert not points["second"].supplied_state_membership_certified


def test_raw_complete_state_and_weights_precede_all_scientific_work(
    monkeypatch, points
):
    actual = points["generic"]
    arguments = dict(
        forms=actual.forms,
        phase_lifts=actual.phase_lifts,
        first_new_edge_conductances=points["weights"][0],
        second_new_edge_conductances=points["weights"][1],
    )

    def forbidden(*args, **kwargs):
        pytest.fail("invalid raw input reached geometry, trigonometry or field work")

    invalid = (
        True,
        np.bool_(False),
        None,
        "0",
        float("nan"),
        float("inf"),
        complex(1, 2),
    )
    with monkeypatch.context() as patch:
        for name in (
            "_derive_coupling_successor_geometry",
            "_mixed_phase_sensitive_field",
            "sin",
            "cos",
        ):
            patch.setattr(owner, name, forbidden)
        for name, width in (
            ("forms", 36),
            ("phase_lifts", 36),
            ("first_new_edge_conductances", 63),
            ("second_new_edge_conductances", 63),
        ):
            for bad in invalid:
                for position in (0, width - 1):
                    changed = dict(arguments)
                    changed[name] = tuple(
                        bad if j == position else value
                        for j, value in enumerate(arguments[name])
                    )
                    with pytest.raises((TypeError, ValueError, OverflowError)):
                        owner._observe_mixed_cycle_current(**changed)
            for bad in (
                (0,) * (width - 1),
                (0,) * (width + 1),
                repeat(0),
                {0, 1},
                {0: 0},
                "0" * width,
            ):
                with pytest.raises((TypeError, ValueError)):
                    owner._observe_mixed_cycle_current(**dict(arguments, **{name: bad}))
        for name in ("first_new_edge_conductances", "second_new_edge_conductances"):
            for bad in (Q(3, 4) - Q(1, 10**100), Q(1) + Q(1, 10**100), I(1)):
                changed = (bad,) + arguments[name][1:]
                with pytest.raises((TypeError, ValueError)):
                    owner._observe_mixed_cycle_current(
                        **dict(arguments, **{name: changed})
                    )
    geometry = owner._derive_coupling_successor_geometry(completion="common_sense")
    altered_cycle = (
        (geometry.internal_edges[0][0], geometry.internal_edges[0][1], Q(2)),
    ) + geometry.internal_edges[1:]
    for altered in (
        replace(geometry, internal_edges=altered_cycle),
        replace(geometry, new_edge_pairs=tuple(reversed(geometry.new_edge_pairs))),
        replace(geometry, target_phase_turns=(Q(0),) * 18),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(
                owner, "_derive_coupling_successor_geometry", lambda **kwargs: altered
            )
            patch.setattr(owner, "_mixed_phase_sensitive_field", forbidden)
            with pytest.raises(ValueError):
                owner._observe_mixed_cycle_current(**arguments)


def test_detached_immutable_readings_do_not_authenticate_a_source_or_install_a_law(
    mp, points, record_property
):
    actual = points["generic"]
    arguments = dict(
        forms=np.array([Q(7)] * 36, dtype=object),
        phase_lifts=_state((Q(0),) * 36, (Q(1),) * 36)[36:],
        first_new_edge_conductances=list(points["weights"][0]),
        second_new_edge_conductances=points["weights"][1],
    )
    outsider = owner._observe_mixed_cycle_current(**arguments)
    assert all(value.lo <= 7 <= value.hi for value in outsider.region_mean_forms)
    assert all(value.lo <= 1 <= value.hi for value in outsider.region_mean_phases)
    for name in (
        "supplied_state_authenticated_as_source",
        "supplied_state_membership_certified",
        "collective_derivative_closed_on_readings",
        "new_field_installed",
        "trajectory_evaluated",
        "new_measurement_acquired",
        "physical_current_identified",
    ):
        assert getattr(outsider, name) is False
    assert actual.region_nodes == REGIONS and actual.cycle_edges == CYCLES
    assert (
        actual.full_state_dimension == 72 and actual.physical_capacities == (Q(1),) * 36
    )
    assert actual.internal_divisor == 9
    assert actual.continuity_equation == "x_prime=-div(J)"
    assert actual.phase_continuity_equation == "theta_prime=-div(P)"
    assert actual.current_units == "form per structural tau"
    assert actual.phase_current_units == "phase per structural tau"
    assert actual.cycle_derivative_units == "form per structural tau squared"
    with pytest.raises(FrozenInstanceError):
        actual.cycle_currents = ()
    for name in (
        "source",
        "clock",
        "horizon",
        "observed_current",
        "cycle_basis",
        "noise",
        "external_current",
    ):
        with pytest.raises(TypeError):
            owner._observe_mixed_cycle_current(**dict(arguments, **{name: Q(0)}))
    difference = (
        points["second"].cycle_current_derivatives[0]
        - points["first"].cycle_current_derivatives[0]
    )
    record_property(
        "constructed_witness_da_derivative_separation",
        str(_interval_record(difference)),
    )
    record_property("witness_amplitude", str(AMPLITUDE))
    record_property(
        "nominal_cycle_currents",
        str(tuple(_interval_record(v) for v in points["nominal"].cycle_currents)),
    )
    record_property("acquired_source_state_evaluated", "False")
    record_property("new_trajectories", "0")
