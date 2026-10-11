"""Exact controls for a finite raw-area observation through loaded recovery.

Algebraic state and integrated-row witnesses below are not trajectories or
reserved responses. Local Hessian and decay controls retain the complete
nonlinear phase law; no previous model's relaxation coefficient is imported.
"""

from dataclasses import fields
from fractions import Fraction as Q
import hashlib
from pathlib import Path

import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.mathematics._exact_linear_algebra import (
    exact_matrix_inverse,
    exact_square_matrix_power,
    exact_symmetric_semidefinite,
)
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin
import tnfr.physics._organized_recovery_observation as recovery_owner
from tnfr.physics._organized_recovery_observation import bound_recovery_area_observation
from tnfr.physics._organized_window_observation import bound_loaded_window_observation


@pytest.fixture(scope="module", autouse=True)
def preserve_frozen_evidence_and_producers():
    root = Path(__file__).resolve().parents[2]
    paths = tuple(
        root / name
        for name in (
            "docs/assets/organized_finite_observation/attempt.json",
            "docs/assets/organized_finite_observation/result.json",
            "benchmarks/organized_finite_observation.py",
            "src/tnfr/physics/_organized_finite_observation_source.py",
            "docs/assets/organized_robust_current/attempt.json",
            "docs/assets/organized_robust_current/result.json",
            "benchmarks/organized_robust_current.py",
            "src/tnfr/physics/_organized_robust_current.py",
            "docs/assets/organized_finite_window_design/method-sources-v1.json",
        )
    )
    before = tuple(hashlib.sha256(path.read_bytes()).digest() for path in paths)
    with forbid_sine_regeneration():
        yield
    assert tuple(hashlib.sha256(path.read_bytes()).digest() for path in paths) == before


_EDGES = tuple((i, (i + 1) % 5) for i in range(5))
_PORT = (Q(1), Q(-1), Q(0), Q(0), Q(0))


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def _action(matrix, vector):
    return tuple(_dot(row, vector) for row in matrix)


@pytest.fixture(scope="module")
def cycle():
    matrix = [[Q(0)] * 5 for _ in range(5)]
    for i, j in _EDGES:
        matrix[i][i] += 1
        matrix[j][j] += 1
        matrix[i][j] -= 1
        matrix[j][i] -= 1
    return tuple(map(tuple, matrix))


def _metric(cycle, ratio):
    square = exact_square_matrix_power(cycle, 2)
    return tuple(
        tuple(Q(i == j) + ratio * square[i][j] for j in range(5)) for i in range(5)
    )


@pytest.fixture(scope="module")
def recovery_bound():
    return bound_recovery_area_observation()


@pytest.mark.parametrize("ratio", (Q(1, 100), Q(1, 10)))
def test_loaded_crossfunctional_differentiation_cancels_both_mixed_terms(cycle, ratio):
    metric = _metric(cycle, ratio)
    inverse = exact_matrix_inverse(metric)
    form = tuple(Q(x, 17) for x in (3, -2, 1, -5, 3))
    theta = tuple(Q(x, 23) for x in (2, -1, 3, -4, 0))
    gamma, coupling, sensor_time, sensor = Q(3, 7), Q(1, 64), Q(1, 1024), Q(2, 19)
    phases, values = (0, 1, -1, 2, 0), (Q(0), Q(1), Q(0), Q(-1))
    source = [Q(0)] * 5
    for i, j in _EDGES:
        current = values[(phases[j] - phases[i]) % 4]
        source[i] += current
        source[j] -= current
    # The differential identity holds for arbitrary fixed phase-chart offsets;
    # the actual twist collar needed for its sign is checked separately below.
    q = _action(cycle, form)
    port, mismatch = _dot(_PORT, q), _dot(_PORT, q) - sensor
    sensor_rate = mismatch / sensor_time
    phase_rate = tuple(gamma * value / 4 for value in q)
    form_rate = _action(
        inverse,
        tuple(
            (-value + gamma * force) / 4 - coupling * v * mismatch
            for value, force, v in zip(q, source, _PORT)
        ),
    )
    p = tuple(
        value + coupling * sensor_time * v * sensor
        for value, v in zip(_action(metric, form), _PORT)
    )
    p_rate = tuple(
        value + coupling * sensor_time * v * sensor_rate
        for value, v in zip(_action(metric, form_rate), _PORT)
    )
    assert p_rate == tuple(
        (-value + gamma * force) / 4 for value, force in zip(q, source)
    )
    energy_rate = (
        _dot(_action(metric, q), form_rate)
        - _dot(source, phase_rate)
        + coupling * sensor_time * sensor * sensor_rate
    )
    cross_rate = gamma * (_dot(phase_rate, p) + _dot(theta, p_rate))
    derivative = energy_rate + cross_rate + _dot(theta, phase_rate)
    expected = (
        -_dot(q, q) / 4
        - coupling * mismatch**2
        + gamma**2
        * (
            _dot(q, _action(metric, form))
            + coupling * sensor_time * sensor * port
            + _dot(theta, source)
        )
        / 4
    )
    assert derivative == expected
    assert _dot(theta, phase_rate) != 0
    assert derivative - _dot(theta, phase_rate) != expected
    # Dropping the retained sensor state from p leaves an uncancelled current.
    assert _action(metric, form_rate) != p_rate


@pytest.mark.parametrize("winding", (-1, 1))
def test_nonlinear_local_pressure_has_the_required_dissipative_pairing(cycle, winding):
    kappa, rho = winding * 2 * pi_interval() / 5, Q(1, 10)
    assert cos(kappa + I(-rho, rho)).lo > Q(1, 5)
    theta = tuple(Q(value, 100) for value in (1, -2, 1, 0, 0))
    assert sum(theta) == 0
    source = [I(0) for _ in range(5)]
    for i, j in _EDGES:
        difference = theta[j] - theta[i]
        assert abs(difference) < rho
        current = sin(kappa + difference)
        source[i] += current
        source[j] -= current
    pairing = sum((value * force for value, force in zip(theta, source)), I(0))
    laplacian_energy = _dot(theta, _action(cycle, theta))
    assert pairing.hi <= -laplacian_energy / 5 < 0
    # L>=I on the neutral subspace, verified independently as a matrix bound.
    assert exact_symmetric_semidefinite(
        tuple(
            tuple(cycle[i][j] - Q(i == j) + Q(1, 5) for j in range(5)) for i in range(5)
        )
    )


def test_own_law_decay_constants_follow_from_coercivity_and_sensor_loading():
    gamma = 1 / (1023 * pi_interval())
    gamma_lower, gamma_upper = Q(1, 4096), Q(1, 3000)
    assert gamma_lower < gamma.lo < gamma.hi < gamma_upper
    wmax, coupling, sensor_time = Q(13, 5), Q(1, 64), Q(1, 1024)
    # Young bounds on the two cross terms leave F>=U²/4+theta²/2+k*tau*z²/4.
    phase_cross_allowance = gamma_upper**2 * (wmax**2 + 4 * coupling * sensor_time)
    assert Q(3, 5) - phase_cross_allowance > Q(1, 2)
    assert 2 * wmax + Q(1, 4) < 6
    assert Q(5, 2) + phase_cross_allowance < 3
    # z=y-d and y²<=2||q||² give F<=7q²+3theta²+2k*tau*d².
    assert 6 + 4 * coupling * sensor_time < 7
    assert Q(1, 4) - gamma_upper**2 * (wmax + 3 * coupling * sensor_time) / 4 > Q(1, 8)
    assert coupling - gamma_upper**2 * coupling * sensor_time / 8 > coupling / 2
    rate = Q(1, 2**32)
    assert 7 * rate < Q(1, 8)
    assert 3 * rate < gamma_lower**2 / 20
    assert 2 * coupling * sensor_time * rate < coupling / 2
    # exp(1)>1+1+1/2>2: each complete structural block contracts by <1/2.
    assert 1 + 1 + Q(1, 2) > 2


@pytest.mark.parametrize("ratio", (Q(1, 100), Q(1, 10)))
def test_raw_area_identities_retain_initial_and_terminal_states(cycle, ratio):
    metric = _metric(cycle, ratio)
    inverse = exact_matrix_inverse(metric)
    initial_form = tuple(Q(value, 17) for value in (3, -2, 1, -5, 3))
    initial_phase = tuple(Q(value, 29) for value in (2, -3, 4, -1, -2))
    integrated_form = tuple(Q(value, 23) for value in (-1, 2, 4, -2, -3))
    initial_sensor, terminal_sensor, sensor_time, coupling = (
        Q(3, 11),
        Q(-2, 13),
        Q(1, 1024),
        Q(1, 64),
    )
    amplitude, pulse_time, gamma = Q(1, 16), Q(1, 1024), Q(3, 7)
    integrated_q = _action(cycle, integrated_form)
    phase_increment = tuple(gamma * value / 4 for value in integrated_q)
    terminal_phase = tuple(a + b for a, b in zip(initial_phase, phase_increment))
    sensor_increment = terminal_sensor - initial_sensor
    raw_area = _dot(_PORT, integrated_q) - sensor_time * sensor_increment
    phase_area = (
        4
        * _dot(_PORT, tuple(a - b for a, b in zip(terminal_phase, initial_phase)))
        / gamma
        - sensor_time * sensor_increment
    )
    assert raw_area == phase_area
    assert raw_area != 4 * _dot(_PORT, terminal_phase) / gamma
    # Impose the integrated linear form row, not a sampled trajectory. Its
    # solution determines the endpoint and retains exactly the same sensor jump.
    form_increment = _action(
        inverse,
        tuple(
            -value / 4
            + amplitude * pulse_time * v
            - coupling * v * sensor_time * sensor_increment
            for value, v in zip(integrated_q, _PORT)
        ),
    )
    terminal_form = tuple(a + b for a, b in zip(initial_form, form_increment))
    linear_area = (
        8 * amplitude * pulse_time
        + 4
        * _dot(
            _PORT,
            _action(metric, tuple(a - b for a, b in zip(initial_form, terminal_form))),
        )
        - (1 + 8 * coupling) * sensor_time * sensor_increment
    )
    assert raw_area == linear_area
    assert linear_area != raw_area + 8 * coupling * sensor_time * sensor_increment


def test_fixed_finite_declaration_separates_actual_permitted_records(recovery_bound):
    bound = recovery_bound
    pulse = bound_loaded_window_observation(observation_error=0)
    assert bound.pulse_energy_upper == pulse.energy_upper < Q(1, 2**16)
    assert bound.identity_certified and bound.response_separated
    assert bound.response_gap > Q(1, 2**14)
    assert bound.full_area_bounds[1] < bound.linear_area_bounds[0]
    assert bound.response_gap == bound.linear_area_bounds[0] - bound.full_area_bounds[1]
    assert bound.decay_exponent == 256
    assert bound.full_state_squared_norm_upper < Q(1, 2 ** (bound.decay_exponent + 7))
    assert bound.linear_state_squared_norm_upper < bound.full_state_squared_norm_upper
    assert bound.full_state_squared_norm_upper <= bound.endpoint_norm_upper**2
    assert (
        bound.sensor_endpoint_upper**2
        * bound.sensor_coupling
        * bound.sensor_time_constant
        >= bound.full_state_squared_norm_upper
    )
    assert 0 < bound.full_terminal_remainder_upper < Q(1, 2**116)
    assert 0 < bound.linear_terminal_remainder_upper < Q(1, 2**127)
    assert bound.total_horizon == bound.pulse_duration + bound.recovery_duration
    center = 8 * bound.amplitude * bound.pulse_duration
    full_phase_prior = 6 * 4096 * bound.preparation_radius
    assert center - full_phase_prior == Q(1, 2**13)
    remaining_initial = bound.full_asymptotic_area_bounds[1] - full_phase_prior
    remaining_initial += center - bound.linear_asymptotic_area_bounds[0]
    assert remaining_initial < Q(1, 2**22)
    assert 2 * bound.area_error == Q(1, 2**17)
    assert (
        bound.full_terminal_remainder_upper + bound.linear_terminal_remainder_upper
        < Q(1, 2**115)
    )
    assert Q(1, 2**13) - Q(1, 2**22) - Q(1, 2**17) - Q(1, 2**115) > Q(1, 2**14)
    # Short-window response ambiguity remains a separate, unchanged result.
    assert bound_loaded_window_observation().phase_blind_records_overlap


def test_recovery_rebuilds_the_fixed_pulse_without_inheriting_changed_defaults(
    monkeypatch,
):
    original = recovery_owner.bound_loaded_window_observation
    declarations = []

    def observe_declaration(**kwargs):
        declarations.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(
        recovery_owner, "bound_loaded_window_observation", observe_declaration
    )
    short = recovery_owner.bound_recovery_area_observation(recovery_duration=0)
    long = recovery_owner.bound_recovery_area_observation(recovery_duration=Q(2**40))
    expected = {
        "response_lower": Q(1, 100),
        "response_upper": Q(1, 10),
        "amplitude": Q(1, 16),
        "horizon": Q(1, 1024),
        "preparation_radius": Q(1, 2**26),
        "sensor_preparation_radius": Q(1, 2**26),
        "sensor_coupling": Q(1, 64),
        "observation_error": Q(0),
    }
    assert declarations == [expected, expected]
    assert short.pulse_duration == long.pulse_duration == expected["horizon"]
    assert (
        short.sensor_time_constant == long.sensor_time_constant == expected["horizon"]
    )
    assert short.pulse_energy_upper == long.pulse_energy_upper


@pytest.mark.parametrize("blocks", (0, 1, 2, 255, 256, 4096))
def test_dyadic_decay_budget_retains_fractional_block_and_parity(blocks):
    block = Q(2**32)
    exact = bound_recovery_area_observation(recovery_duration=blocks * block)
    partial = bound_recovery_area_observation(
        recovery_duration=blocks * block + block / 3
    )
    assert exact.decay_exponent == partial.decay_exponent == blocks
    assert exact.decay_factor_upper == partial.decay_factor_upper
    assert exact.full_area_bounds == partial.full_area_bounds
    assert exact.endpoint_norm_upper**2 >= exact.full_state_squared_norm_upper
    assert exact.sensor_time_constant == partial.sensor_time_constant == Q(1, 1024)
    if blocks:
        preceding = bound_recovery_area_observation(
            recovery_duration=blocks * block - 1
        )
        assert preceding.decay_factor_upper == 2 * exact.decay_factor_upper
        assert exact.endpoint_norm_upper <= preceding.endpoint_norm_upper


def test_weak_horizon_and_large_error_are_unavailable_certificates_not_exceptions():
    short = bound_recovery_area_observation(recovery_duration=0)
    assert short.identity_certified and not short.response_separated
    assert short.response_gap < 0
    noisy = bound_recovery_area_observation(area_error=1)
    assert noisy.identity_certified and not noisy.response_separated
    assert noisy.full_terminal_remainder_upper > 0


def test_restoring_countermodel_has_same_zero_net_area_without_sine_identification(
    cycle,
    recovery_bound,
):
    kappa, shift = 2 * pi_interval() / 5, Q(1, 100)
    stiffness = cos(kappa)
    assert stiffness.lo > Q(1, 5)
    # On the admitted local lift, V_quad=cos(kappa)*theta.L.theta/2 has
    # the same equilibrium and tangent as the sine well, but another force.
    sine_force = -2 * stiffness * sin(I(shift))
    quadratic_force = -2 * stiffness * shift
    assert (sine_force - quadratic_force).lo > 0
    theta = tuple(shift * (Q(i == 0) - Q(1, 5)) for i in range(5))
    assert sum(theta) == 0
    assert _dot(theta, _action(cycle, theta)) > 0
    # Each oriented twist edge has the same turn modulo one, so its sine
    # derivative is cos(kappa) times its gap rate. Sum those coefficients
    # independently, retaining the closing edge's whole-turn offset.
    turns = tuple(Q(i, 5) for i in range(5))
    sine_jacobian_coefficient = [Q(0)] * 5
    for i, j in _EDGES:
        assert (turns[j] - turns[i]) % 1 == Q(1, 5)
        gap_rate = theta[j] - theta[i]
        sine_jacobian_coefficient[i] += gap_rate
        sine_jacobian_coefficient[j] -= gap_rate
    quadratic_jacobian_coefficient = tuple(-value for value in _action(cycle, theta))
    assert tuple(sine_jacobian_coefficient) == quadratic_jacobian_coefficient
    # All other full-row derivatives are unchanged; the same response matrix
    # maps these identical phase blocks, despite their distinct nonlinear fields.
    inverse = exact_matrix_inverse(_metric(cycle, Q(1, 100)))
    assert _action(inverse, sine_jacobian_coefficient) == _action(
        inverse, quadratic_jacobian_coefficient
    )
    # Matched initial terms cancel exactly. Even opposite terminal errors
    # cannot resolve these restoring laws under the declared area error.
    tail = recovery_bound.full_terminal_remainder_upper
    assert 2 * tail < 2 * recovery_bound.area_error
    sine_area, quadratic_area = -tail, tail
    common_record = (sine_area + quadratic_area) / 2
    assert abs(common_record - sine_area) <= recovery_bound.area_error
    assert abs(common_record - quadratic_area) <= recovery_bound.area_error


def test_quadratic_restoring_integrated_response_differs_from_phase_blind_dc(cycle):
    # Independently build both complete generators on exact neutral coordinates
    # (first four values, with the fifth minus their sum) and retained sensor.
    # Their passive storage proves decay; inversion evaluates the integrated
    # row identity, not a trajectory or the nonlinear model's reserved response.
    inverse = exact_matrix_inverse(_metric(cycle, Q(1, 10)))
    gamma, stiffness, coupling, sensor_time = Q(3, 7), Q(3, 10), Q(1, 64), Q(1, 1024)
    delivered = Q(1, 16) * Q(1, 1024)

    def neutral(first_four):
        return tuple(first_four) + (-sum(first_four),)

    def generator(with_phase):
        dimension = 9 if with_phase else 5
        columns = []
        for index in range(dimension):
            coordinates = tuple(Q(i == index) for i in range(dimension))
            form = neutral(coordinates[:4])
            phase = neutral(coordinates[4:8]) if with_phase else (Q(0),) * 5
            sensor = coordinates[-1]
            q, phase_gradient = _action(cycle, form), _action(cycle, phase)
            mismatch = _dot(_PORT, q) - sensor
            form_rate = _action(
                inverse,
                tuple(
                    (-value - gamma * stiffness * restoring) / 4
                    - coupling * v * mismatch
                    for value, restoring, v in zip(q, phase_gradient, _PORT)
                ),
            )
            column = form_rate[:4]
            if with_phase:
                column += tuple(gamma * value / 4 for value in q[:4])
            columns.append(column + (mismatch / sensor_time,))
        return tuple(
            tuple(columns[j][i] for j in range(dimension)) for i in range(dimension)
        )

    input_form = tuple(delivered * value for value in _action(inverse, _PORT))
    full_generator = generator(True)
    full_input = input_form[:4] + (Q(0),) * 5
    integrated_full = _action(
        exact_matrix_inverse(full_generator), tuple(-value for value in full_input)
    )
    assert _action(full_generator, integrated_full) == tuple(
        -value for value in full_input
    )
    assert integrated_full[:4] == (Q(0),) * 4
    assert integrated_full[-1] == 0
    assert any(integrated_full[4:8])
    assert _action(cycle, neutral(integrated_full[4:8])) == tuple(
        4 * delivered * value / (gamma * stiffness) for value in _PORT
    )
    linear_generator = generator(False)
    linear_input = input_form[:4] + (Q(0),)
    integrated_linear = _action(
        exact_matrix_inverse(linear_generator), tuple(-value for value in linear_input)
    )
    assert _action(linear_generator, integrated_linear) == tuple(
        -value for value in linear_input
    )
    assert integrated_linear[-1] == 8 * delivered > integrated_full[-1]


def test_laboratory_integral_scale_cancels_clock_without_becoming_a_mean(
    recovery_bound,
):
    a0, gamma = Q(3, 7), Q(1, 37)
    # Piecewise constant abstract sensor histories establish the change of
    # variables independently of an endpoint identity or a fitted response.
    pieces = ((Q(2, 13), Q(3, 17)), (Q(5, 19), Q(-2, 23)))
    structural_area = sum(duration * value for duration, value in pieces)
    for capacitance, conductance in ((Q(5, 11), Q(7, 17)), (Q(13, 19), Q(2, 3))):
        clock = 4 * conductance / capacitance
        voltage_scale = a0 * gamma * conductance / capacitance
        physical_area = sum(
            duration / clock * voltage_scale * value for duration, value in pieces
        )
        assert physical_area == a0 * gamma * structural_area / 4
        error_area = voltage_scale * recovery_bound.area_error / clock
        assert error_area == a0 * gamma * recovery_bound.area_error / 4
        lab_horizon = recovery_bound.total_horizon / clock
        allowed_mean_error = error_area / lab_horizon
        assert allowed_mean_error * lab_horizon == error_area
        assert allowed_mean_error != voltage_scale * recovery_bound.area_error


@pytest.mark.parametrize("parameter", ("recovery_duration", "area_error"))
@pytest.mark.parametrize("invalid", (True, float("nan"), float("inf"), "1", -1))
def test_primitive_admission_precedes_decay_arithmetic(parameter, invalid):
    with pytest.raises((TypeError, ValueError)):
        bound_recovery_area_observation(**{parameter: invalid})


def test_closed_arithmetic_budget_rejects_before_large_exponent_allocation():
    with pytest.raises(ValueError, match="4096"):
        bound_recovery_area_observation(recovery_duration=4097 * 2**32)
    with pytest.raises(ValueError, match="4096"):
        bound_recovery_area_observation(recovery_duration=Q(10**100))


def test_fixed_width_fraction_normalization_and_exact_input_boundaries(recovery_bound):
    np = pytest.importorskip("numpy")
    payload = Q(np.int64(2**40), np.int64(1))
    actual = bound_recovery_area_observation(recovery_duration=payload)
    assert actual == recovery_bound
    assert (
        bound_recovery_area_observation(area_error=float(Q(1, 2**18))) == recovery_bound
    )
    with pytest.raises((TypeError, ValueError)):
        bound_recovery_area_observation(recovery_duration=np.bool_(True))
    with pytest.raises(ValueError, match="4096"):
        bound_recovery_area_observation(
            recovery_duration=Q(np.int64(2**62), np.int64(3))
        )
    for field in fields(actual):
        value = getattr(actual, field.name)
        for scalar in value if isinstance(value, tuple) else (value,):
            if isinstance(scalar, Q):
                assert type(scalar.numerator) is type(scalar.denominator) is int
    exact = bound_recovery_area_observation(area_error=0)
    assert exact.response_separated and exact.response_gap > recovery_bound.response_gap
