"""Static controls for regular gain-dependent Emission stopping maps.

These checks retain every target phase, not only the child form mean. The
45-node witness is a supplied leaf state, not the acquired source or a new
trajectory. Auxiliary exact flows test the regularity boundary, without
claiming that their laws or stopping mechanisms are realized by TNFR.
"""

from fractions import Fraction as Q

import mpmath
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._rational_interval import I
from tnfr.physics import _organized_source_capture as capture
from tnfr.physics import _organized_source_port as owner

ZERO = (Q(0),) * 45
OFFSETS = (Q(0),) * 36 + (Q(1, 8),) * 9
CURRENT_COEFFICIENTS = (Q(1),) * 9 + (Q(0),) * 27 + (Q(-1),) * 9


@pytest.fixture(scope="module", autouse=True)
def no_response_or_source_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("Stopping controls cannot execute or rebuild a source")

    with forbid_organized_response_evaluation(), pytest.MonkeyPatch.context() as patch:
        for object_, name in (
            (runtime, "step"),
            (runtime, "run"),
            (runtime, "_update_nodes"),
            (integrators.DefaultIntegrator, "integrate"),
            (capture, "_assess_organized_source_capture"),
            (capture, "_assess_organized_source_attachment"),
            (owner, "_assess_organized_source_attachment"),
            (owner, "_assess_organized_source_emission"),
        ):
            patch.setattr(object_, name, forbidden)
        yield


@pytest.fixture(scope="module")
def geometry():
    return owner._derive_organized_source_port()


@pytest.fixture(scope="module")
def witness():
    return owner._observe_organized_source_port(form=ZERO, phase_offsets=OFFSETS)


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 80
    return context


def _mp(mp, value):
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, bounds, expected):
    assert _mp(mp, bounds.lo) <= expected <= _mp(mp, bounds.hi)


def _laplacian(edges, values):
    result = [Q(0) for _ in values]
    for left, right, weight in edges:
        difference = weight * (values[left] - values[right])
        result[left] += difference
        result[right] -= difference
    return tuple(result)


def test_nine_target_phase_constraints_have_rank_nine_on_the_form_leaf(geometry):
    edges = geometry.weighted_edges
    assert len(edges) == 90
    assert geometry.clock_factor == Q(1, 4)
    assert geometry.held_capacities == (Q(1),) * 45
    assert geometry.gamma_bounds.lo > 0
    # Child0 sees child1 but not child3. Each direction has zero form sum;
    # its target Laplacian is one negative basis vector, an exact right inverse.
    columns = []
    for slot in range(9):
        direction = [Q(0)] * 45
        direction[9 + slot], direction[27 + slot] = Q(1), Q(-1)
        assert sum(direction) == 0
        target_gradient = _laplacian(edges, direction)[:9]
        assert target_gradient == tuple(-Q(index == slot) for index in range(9))
        columns.append(target_gradient)
    for row in range(9):
        assert tuple(column[row] for column in columns) == tuple(
            -Q(index == row) for index in range(9)
        )


def test_zero_mean_phase_rate_does_not_preserve_every_target_phase(geometry):
    amplitude = Q(1, 64)
    form = [Q(0)] * 45
    form[9], form[27] = amplitude, -amplitude
    form[10], form[28] = -amplitude, amplitude
    report = owner._observe_organized_source_port(
        form=tuple(form), phase_offsets=OFFSETS
    )
    gradient = _laplacian(geometry.weighted_edges, form)
    assert sum(form) == 0 and sum(OFFSETS) == Q(9, 8)
    assert report.form_gradient == gradient
    assert gradient[:9] == (-amplitude, amplitude) + (Q(0),) * 7
    assert sum(gradient[:9]) == 0
    assert report.child_phase_tau_rate_bounds[0].contains(0)
    assert report.phase_tau_rate_bounds[0].hi < 0
    assert report.phase_tau_rate_bounds[1].lo > 0
    assert all(rate == I(0) for rate in report.phase_tau_rate_bounds[2:9])
    # Multiplying by any positive finite stopping derivative cannot turn
    # either nonzero constituent phase rate into the required zero.
    for derivative in (Q(1, 4), Q(1), Q(4)):
        assert (derivative * report.phase_tau_rate_bounds[0]).hi < 0
        assert (derivative * report.phase_tau_rate_bounds[1]).lo > 0


def test_positive_common_target_current_passes_only_the_first_order_conditions(
    witness, mp
):
    assert sum(witness.form) == 0
    assert sum(witness.phase_offsets) == Q(9, 8)
    assert witness.form_gradient == ZERO
    assert all(rate == I(0) for rate in witness.phase_tau_rate_bounds)
    gamma, sine = 1 / (1023 * mp.pi), mp.sin(mp.mpf(1) / 8)
    expected_rate = gamma * sine / 4
    stopping_derivative = 1 / expected_rate
    assert sine > 0 and stopping_derivative > 0
    for node, coefficient in enumerate(CURRENT_COEFFICIENTS):
        current = _mp(mp, coefficient) * sine
        _contains(mp, witness.phase_current_bounds[node], current)
        _contains(mp, witness.form_tau_rate_bounds[node], gamma * current / 4)
    for node in range(9):
        assert witness.phase_current_bounds[node].lo > 0
        assert witness.form_tau_rate_bounds[node].lo > 0
        _contains(mp, witness.form_tau_rate_bounds[node], 1 / stopping_derivative)
    # The actual fine observer's enclosure is checked against the independent
    # cancellation of internal cycle currents and the donor's opposite current.
    assert sum(CURRENT_COEFFICIENTS) == 0


def test_first_order_witness_has_an_unavoidable_second_order_phase_change(witness, mp):
    edges = witness.geometry.weighted_edges
    coefficients = _laplacian(edges, CURRENT_COEFFICIENTS)
    assert coefficients[:9] == (Q(4),) * 9
    # Differentiate the actual linear phase row: theta''=(gamma/4)L*x'.
    # Since theta'=0 here, no phase-current Jacobian term is needed.
    differentiated = owner._observe_organized_source_acceleration(
        form=witness.form, phase_offsets=witness.phase_offsets
    )
    acceleration = differentiated.phase_tau_acceleration_bounds
    gamma, sine = 1 / (1023 * mp.pi), mp.sin(mp.mpf(1) / 8)
    expected_acceleration = gamma**2 * sine / 4
    for node in range(9):
        assert acceleration[node].lo > 0
        _contains(mp, acceleration[node], expected_acceleration)
        inverse_rate = 1 / witness.form_tau_rate_bounds[node]
        stopped_coefficient = inverse_rate**2 * acceleration[node] / 2
        assert stopped_coefficient.lo > 0
        _contains(mp, stopped_coefficient, 2 / sine)
    # T(b)=a*b+o(b) suffices: the b² coefficient is theta''*a²/2=2/sin(1/8).
    # A second-order time correction multiplies theta'=0 and cannot cancel it.
    # This is a derivative obstruction, not an evaluated finite trajectory.


def test_auxiliary_regular_stopping_matches_target_and_chain_rule(mp):
    # Analytic x'=x, passive phi'=0 on x>0. This independent model admits
    # T(x,b)=log(1+b/x); its derivative at zero gain is finite and positive.
    phase = Q(1, 40)
    for initial in (Q(1, 8), Q(1, 4), Q(1, 2)):
        x = _mp(mp, initial)

        def time(gain):
            return mp.log(1 + gain / x)

        def target(gain):
            return x * mp.exp(time(gain)), _mp(mp, phase)

        assert time(0) == 0
        first = mp.diff(time, 0)
        second = mp.diff(time, 0, 2)
        assert first > 0
        assert abs(first - 1 / x) < mp.mpf("1e-70")
        assert abs(first * x - 1) < mp.mpf("1e-70")
        assert abs(first**2 * x + second * x) < mp.mpf("1e-70")
        assert mp.diff(lambda gain: target(gain)[1], 0) == 0
        for gain in (Q(1, 64), Q(1, 32), Q(1, 8)):
            represented = _mp(mp, gain)
            assert time(represented) > 0
            form, endpoint_phase = target(represented)
            assert abs(form - x - represented) < mp.mpf("1e-70")
            assert endpoint_phase == _mp(mp, phase)
    # These ideal-real endpoint identities do not assert that an arbitrarily
    # small gain survives binary64 assignment or that native grammar runs.


def test_singular_square_root_stopping_evades_the_finite_derivative_condition():
    # Analytic auxiliary field x'=2v, v'=1, phi'=0 from x=v=0. Only (x,phi)
    # is observed; collateral v may change. Its exact solution is (t²,t,phi).
    phase = Q(1, 40)

    def state(time):
        return time**2, time, phase

    for time in (Q(0), Q(1, 4), Q(1, 2)):
        h = Q(1, 128)
        central_derivative = tuple(
            (after - before) / (2 * h)
            for before, after in zip(state(time - h), state(time + h))
        )
        assert central_derivative == (2 * time, Q(1), Q(0))
    ratios = []
    for exponent in (1, 2, 3, 4):
        gain, time = Q(1, 4**exponent), Q(1, 2**exponent)
        endpoint = state(time)
        assert endpoint == (gain, time, phase)
        assert endpoint[0] - state(0)[0] == gain
        assert endpoint[2] == state(0)[2]
        ratios.append(time / gain)
        assert time / gain == 2**exponent
    assert ratios == sorted(set(ratios))
    # T(b)/b=1/sqrt(b) diverges as b decreases to zero; no finite a can turn
    # the initial target form rate zero into unit gain derivative.
    assert state(0)[1] == 0
    for finite_derivative in (Q(1, 4), Q(1), Q(4)):
        assert finite_derivative * (2 * state(0)[1]) != 1
