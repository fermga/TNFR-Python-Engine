"""Complete-field stages and admission for the opt-in shared RK4 kernel.

These independent polynomial and oscillator controls contain no maintained
contact, reserved response, operator event or manufactured temporal record.
"""

import sys
from fractions import Fraction as Q
from math import inf, isfinite, nan, ulp

import pytest

from tnfr.dynamics.integrators import rk4_coupled_step


def test_complete_harmonic_rows_match_the_independent_stability_polynomial():
    calls = []

    def oscillator(time, state):
        assert type(state) is tuple
        assert all(type(value) is float for value in state)
        calls.append((time, state))
        x, velocity = state
        return velocity, -x

    initial = (1.0, 0.0)
    h = Q(1, 8)
    result = rk4_coupled_step(initial, oscillator, t=7, dt=h)

    expected = (1 - h**2 / 2 + h**4 / 24, -h + h**3 / 6)
    for actual, exact in zip(result, expected, strict=True):
        assert abs(Q(actual) - exact) <= Q(ulp(float(exact)))
    assert [time for time, _ in calls] == [7.0, 7 + 1 / 16, 7 + 1 / 16, 7 + 1 / 8]
    assert calls[1][1] == (1.0, -1 / 16)
    assert calls[2][1] == (1 - 1 / 256, -1 / 16)
    assert calls[3][1] == (1 - 1 / 128, -(1 - 1 / 256) / 8)
    assert initial == (1.0, 0.0)
    assert type(result) is tuple


def test_non_autonomous_coupled_polynomial_has_exact_analytic_solution():
    # y'=t and x'=y imply y(t+h)=y+t*h+h^2/2 and a cubic x.
    t, h = Q(2), Q(1, 4)
    expected = (1 + 3 * h + t * h**2 / 2 + h**3 / 6, 3 + t * h + h**2 / 2)
    result = rk4_coupled_step((1, 3), lambda time, state: (state[1], time), t=t, dt=h)
    for actual, exact in zip(result, expected, strict=True):
        assert abs(Q(actual) - exact) <= Q(ulp(float(exact)))


def test_no_clipping_phase_wrap_or_graph_dispatch_is_inferred():
    initial = (-2, 7)
    result = rk4_coupled_step(initial, lambda time, state: (-1, 2), t=-1, dt=1 / 2)
    assert result == (-2.5, 8.0)
    assert initial == (-2, 7)


def test_zero_rate_keeps_every_coordinate_and_normalizes_signed_zero():
    assert rk4_coupled_step(
        (-0.0, -2.0, 1 / 4), lambda time, state: (0, 0, 0), t=0, dt=1
    ) == (0.0, -2.0, 0.25)


@pytest.mark.parametrize("state", ((), [], [1.0], "x", 1, None))
def test_state_requires_a_nonempty_immutable_tuple_before_callbacks(state):
    def forbidden(*args):
        pytest.fail("invalid initial state must reject before callback")

    with pytest.raises(TypeError, match="nonempty tuple"):
        rk4_coupled_step(state, forbidden, t=0, dt=1)


@pytest.mark.parametrize("bad", (True, 1j, "1", None, nan, inf, -inf, Q(1, 2**2000)))
def test_state_scalar_admission_precedes_callback(bad):
    def forbidden(*args):
        pytest.fail("invalid scalar must reject before callback")

    with pytest.raises((TypeError, ValueError)):
        rk4_coupled_step((1, bad), forbidden, t=0, dt=1)


@pytest.mark.parametrize(
    "start, step",
    (
        (True, 1),
        (nan, 1),
        (inf, 1),
        (Q(1, 2**2000), 1),
        (0, True),
        (0, 0),
        (0, -1),
        (0, nan),
        (0, inf),
        (0, Q(1, 2**2000)),
        (0, ulp(0.0)),
        (2**60, 1),
        (1.0, ulp(1.0)),
        (sys.float_info.max, sys.float_info.max),
    ),
)
def test_clock_admission_and_distinct_stage_times_precede_callback(start, step):
    def forbidden(*args):
        pytest.fail("invalid clock must reject before callback")

    with pytest.raises((TypeError, ValueError)):
        rk4_coupled_step((0,), forbidden, t=start, dt=step)


def test_non_callable_field_rejects():
    with pytest.raises(TypeError, match="callable"):
        rk4_coupled_step((0,), None, t=0, dt=1)


@pytest.mark.parametrize(
    "rates",
    (
        [],
        [0, 0],
        (),
        (0,),
        (0, 0, 0),
        (0, True),
        (0, nan),
        (0, inf),
        (0, Q(1, 2**2000)),
    ),
)
def test_rhs_shape_and_scalars_are_admitted_at_every_stage(rates):
    calls = []

    def field(time, state):
        calls.append((time, state))
        return (1, 2) if len(calls) < 3 else rates

    with pytest.raises((TypeError, ValueError)):
        rk4_coupled_step((0, 0), field, t=0, dt=1)
    assert len(calls) == 3


def test_rhs_exception_is_not_replaced_or_retried():
    calls = []

    def field(time, state):
        calls.append(time)
        if len(calls) == 2:
            raise RuntimeError("declared field domain")
        return (1,)

    with pytest.raises(RuntimeError, match="declared field domain"):
        rk4_coupled_step((1,), field, t=0, dt=1)
    assert calls == [0.0, 0.5]


def test_nonfinite_stage_output_rejects_without_later_callbacks():
    calls = []

    def field(time, state):
        calls.append(time)
        return (sys.float_info.max,)

    with pytest.raises(ValueError, match="finite"):
        rk4_coupled_step((sys.float_info.max,), field, t=0, dt=2)
    assert calls == [0.0]


def test_stage_cannot_swallow_a_nonzero_subnormal_update_into_zero():
    calls = []

    def field(time, state):
        calls.append(time)
        return (ulp(0.0),)

    with pytest.raises(ValueError, match="nonzero represented-input"):
        rk4_coupled_step((0,), field, t=0, dt=1)
    assert calls == [0.0]


def test_final_quadrature_cannot_swallow_a_nonzero_subnormal_result():
    calls = []

    def field(time, state):
        calls.append(time)
        return (ulp(0.0) if len(calls) == 4 else 0.0,)

    with pytest.raises(ValueError, match="endpoint loses a nonzero"):
        rk4_coupled_step((0,), field, t=0, dt=1)
    assert len(calls) == 4


def test_shared_quadrature_recovers_finite_output_despite_weighted_sum_overflow():
    rate = sys.float_info.max
    result = rk4_coupled_step((0,), lambda time, state: (rate,), t=0, dt=1 / 8)
    assert isfinite(result[0])
    assert result == (float(Q(rate) / 8),)
