"""Independent full54 field and common-source observation controls.

Sources are fresh synthetic boxes, not the reserved acquired family. The
tests exercise complete rows and observation arithmetic without formation
assessments or replay of any retained producer or response.
"""

from fractions import Fraction as Q

import mpmath
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.physics import relational_sine_class_readout as owner
from tnfr.physics.phase_cycle_geometry import _derive
from tnfr.physics.relational_sine_two_port_readout import _full_sine_field


@pytest.fixture(scope="module")
def graph():
    nodes = tuple(range(27))
    edges = tuple(
        (9 * component + local, 9 * component + (local + 1) % 9)
        for component in range(3)
        for local in range(9)
    ) + ((4, 13), (13, 22))
    degrees = tuple(sum(node in edge for edge in edges) for node in nodes)
    return nodes, edges, degrees


@pytest.fixture(scope="module")
def full_field(graph):
    nodes, edges, degrees = graph
    model = RelationalExchangeModel(
        1, epi_weight=Q(1023, 1024), phase_weight=Q(1, 1024), phase_domain="regular"
    )
    canonical_edges = tuple(sorted(tuple(sorted(edge)) for edge in edges))
    return _full_sine_field(model, _derive(nodes, canonical_edges), degrees)


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 75
    return context


def _mp(value, mp):
    return mp.mpf(value.numerator) / value.denominator


def _contains_mp(interval, value, mp):
    return _mp(interval.lo, mp) <= value <= _mp(interval.hi, mp)


def _independent_rates(graph, state, gamma, sine):
    _, edges, degrees = graph
    x, theta = state[:27], state[27:]
    ax, current = [0] * 27, [0] * 27
    for i, j in edges:
        difference = x[i] - x[j]
        flow = sine(theta[j] - theta[i])
        ax[i] += difference / degrees[i]
        ax[j] -= difference / degrees[j]
        current[i] += flow / degrees[i]
        current[j] -= flow / degrees[j]
    return tuple(-v + gamma * f for v, f in zip(ax, current)) + tuple(
        gamma * value for value in ax
    )


def test_all54_rates_keep_both_clock_rows_and_nonregular_degrees(graph, full_field, mp):
    _, _, degrees = graph
    flow, domain = full_field
    form = tuple(Q((7 * i + 3) % 17 - 8, 101) for i in range(27))
    phase = tuple(Q((5 * i + 2) % 19 - 9, 23) + 7 * (i % 3 - 1) for i in range(27))
    raw = form + phase
    state = tuple(_mp(value, mp) for value in raw)
    exact = _independent_rates(graph, state, 1 / (1023 * mp.pi), mp.sin)
    bounded = flow(tuple(I(value) for value in raw))
    assert len(bounded) == 54
    assert all(
        _contains_mp(interval, value, mp) for interval, value in zip(bounded, exact)
    )
    assert degrees[4] == 3 and degrees[13] == 4 and degrees[22] == 3
    assert domain(tuple(I(-100, 100) for _ in range(54))) == (Q(1),)
    for channel in range(2):
        charge_rate = sum(
            (degree * bounded[27 * channel + i] for i, degree in enumerate(degrees)),
            I(0),
        )
        assert charge_rate.contains(0)
        assert abs(
            sum(degree * exact[27 * channel + i] for i, degree in enumerate(degrees))
        ) < mp.mpf("1e-70")
    # Omitting the fast-clock conversion from either row changes this state.
    loss = mp.mpf(1023) / 1024
    assert max(abs(value * (1 - loss)) for value in exact[:27]) > mp.mpf("1e-6")
    assert max(abs(value * (1 - loss)) for value in exact[27:]) > mp.mpf("1e-8")


def test_full54_jet_directional_derivative_matches_independent_edge_differentiation(
    graph, full_field, mp
):
    _, edges, degrees = graph
    flow, _ = full_field
    raw = tuple(Q((3 * i + 1) % 23 - 11, 107) for i in range(54))
    direction = tuple(Q((7 * i + 2) % 17 - 8, 109) for i in range(54))
    state = tuple(_mp(value, mp) for value in raw)
    tangent = tuple(_mp(value, mp) for value in direction)
    ax, sine_rate = [mp.mpf(0)] * 27, [mp.mpf(0)] * 27
    for i, j in edges:
        form_difference = tangent[i] - tangent[j]
        phase_rate = mp.cos(state[27 + j] - state[27 + i]) * (
            tangent[27 + j] - tangent[27 + i]
        )
        ax[i] += form_difference / degrees[i]
        ax[j] -= form_difference / degrees[j]
        sine_rate[i] += phase_rate / degrees[i]
        sine_rate[j] -= phase_rate / degrees[j]
    gamma = 1 / (1023 * mp.pi)
    expected = tuple(
        -value + gamma * rate for value, rate in zip(ax, sine_rate)
    ) + tuple(gamma * value for value in ax)
    bounded = flow(
        tuple(Jet((I(value), I(rate))) for value, rate in zip(raw, direction))
    )
    assert all(
        _contains_mp(jet.coeffs[1], value, mp) for jet, value in zip(bounded, expected)
    )


def test_acquired_norm_family_has_an_outer_box_not_an_acquired_corner_certificate(mp):
    eps = Q(1, 10**32)
    pi = pi_interval()
    form_box = (I(-eps, eps),) * 27
    phase_box = tuple(
        2 * pi * (2 if 9 <= i < 18 else 1) * Q(i % 9 - 4, 9) + I(-eps, eps)
        for i in range(27)
    )
    residuals = tuple(Q(i % 9 - 4, 20) * eps for i in range(27))
    assert all(sum(residuals[9 * part : 9 * part + 9], Q(0)) == 0 for part in range(3))
    assert all(
        sum((x**2 for x in residuals[9 * part : 9 * part + 9]), Q(0)) <= eps**2
        for part in range(3)
    )
    for i, residual in enumerate(residuals):
        target = 2 * mp.pi * (2 if 9 <= i < 18 else 1) * (i % 9 - 4) / 9
        assert form_box[i].contains(residual)
        assert _contains_mp(phase_box[i], target + _mp(residual, mp), mp)
    # This admitted outer-box point violates both original component premises.
    assert all(interval.contains(eps) for interval in form_box)
    assert 9 * eps != 0 and 9 * eps**2 > eps**2


def _polynomial_field(*_):
    def flow(state):
        zero = state[0] * 0
        result = [zero] * 54
        result[22] = state[4] * state[4]
        return tuple(result)

    return flow, lambda _: (Q(1),)


def _polynomial_arguments(shift=Q(0), *, delay=Q(1, 5)):
    form = [(Q(0), Q(0))] * 27
    form[4] = (Q(1, 7), Q(1, 7))
    form[22] = (10 + shift, 11 + shift)
    return dict(
        initial_form_bounds=tuple(form),
        initial_phase_bounds=((Q(0), Q(0)),) * 27,
        first_probe_amplitude=Q(1, 6),
        second_probe_amplitude=Q(-1, 9),
        delay=delay,
        total_duration=Q(3, 5),
        time_step=Q(1, 5),
        order=2,
        max_steps=16,
    )


def test_symbolic_suffix_cancellation_retains_wide_shared_prefix_without_reset(
    monkeypatch,
):
    monkeypatch.setattr(owner, "_full_sine_field", _polynomial_field)
    arguments = _polynomial_arguments()
    original = owner.bound_sine_class_four_history_readout(**arguments)
    shifted = owner.bound_sine_class_four_history_readout(
        **_polynomial_arguments(Q(13, 7))
    )
    assert original.admitted and shifted.admitted
    a, b = arguments["first_probe_amplitude"], arguments["second_probe_amplitude"]
    duration = arguments["total_duration"] - arguments["delay"]
    expected_mixed = 2 * a * b * duration
    assert original.mixed_readout_bounds.contains(expected_mixed)
    assert original.mixed_readout_bounds == shifted.mixed_readout_bounds
    assert original.mixed_readout_bounds.width < Q(1, 10**30)
    assert original.raw_endpoint_mixed_bounds.width >= 4
    for index, (first, second) in enumerate(original.history_event_amplitudes):
        expected = (Q(1, 7) + first + second) ** 2 * duration
        assert original.suffix_receiver_increment_bounds[index].contains(expected)
        suffix = original.segments[index + 2]
        parent = original.segments[suffix.parent_segment_index]
        assert suffix.pre_event_box == parent.final_state_bounds
        assert suffix.initial_box[22] == parent.final_state_bounds[22]
        assert suffix.initial_box[27:] == parent.final_state_bounds[27:]


def test_second_event_at_readout_has_exact_zero_mixed_suffix_even_for_wide_sources(
    monkeypatch,
):
    monkeypatch.setattr(owner, "_full_sine_field", _polynomial_field)
    report = owner.bound_sine_class_four_history_readout(
        **_polynomial_arguments(delay=Q(3, 5))
    )
    assert report.admitted
    assert report.mixed_readout_bounds == I(0)
    assert report.suffix_receiver_increment_bounds == (I(0),) * 4
    assert report.raw_endpoint_mixed_bounds.width >= 4
    for suffix in report.segments[2:]:
        assert suffix.steps == ()
        assert suffix.final_state_bounds[22] == suffix.pre_event_box[22]


@pytest.fixture(scope="module")
def nonlinear_control():
    form = tuple(Q((7 * i + 3) % 17 - 8, 1000) for i in range(27))
    phase = tuple(Q((5 * i + 2) % 19 - 9, 19) + 6 * (i % 3 - 1) for i in range(27))
    radius = Q(1, 10**12)
    arguments = dict(
        initial_form_bounds=tuple((v - radius, v + radius) for v in form),
        initial_phase_bounds=tuple((v - radius, v + radius) for v in phase),
        first_probe_amplitude=Q(1, 97),
        second_probe_amplitude=Q(-1, 137),
        delay=Q(1, 2048),
        total_duration=Q(3, 2048),
        time_step=Q(1, 2048),
        order=4,
        max_steps=12,
    )
    return owner.bound_sine_class_four_history_readout(**arguments), form + phase


def test_fresh_full_nonlinear_histories_fit_all54_enclosures_and_common_source_mixed_band(
    graph, nonlinear_control
):
    report, source = nonlinear_control
    assert report.admitted and report.completed_step_count == 10
    gamma = 1 / (1023 * np.pi)

    def rhs(_, state):
        return np.array(_independent_rates(graph, state, gamma, np.sin))

    def advance(state, start, end):
        solution = solve_ivp(
            rhs, (start, end), state, method="DOP853", rtol=3e-13, atol=1e-15
        )
        assert solution.success
        return solution.y[:, -1]

    endpoints = []
    for index, (first, second) in enumerate(report.history_event_amplitudes):
        state = np.array([float(value) for value in source])
        state[4] += float(first)
        state = advance(state, 0.0, float(report.delay))
        pre_receiver = state[22]
        state[4] += float(second)
        state = advance(state, float(report.delay), float(report.total_duration))
        suffix = report.segments[index + 2]
        assert all(
            float(bound.lo) < value < float(bound.hi)
            for bound, value in zip(suffix.final_state_bounds, state)
        )
        increment = report.suffix_receiver_increment_bounds[index]
        assert float(increment.lo) < state[22] - pre_receiver < float(increment.hi)
        endpoints.append(state[22])
    actual_mixed = sum(
        sign * value
        for sign, value in zip(report.mixed_readout_coefficients, endpoints)
    )
    assert (
        float(report.mixed_readout_bounds.lo)
        < actual_mixed
        < float(report.mixed_readout_bounds.hi)
    )
