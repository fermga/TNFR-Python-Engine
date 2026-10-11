"""Independent controls for the loaded C5 sensor and finite-window admission.

These tests evaluate algebra and analytical error bounds, never trajectories,
reserved responses or instrument performance. Rational gamma values below
exercise identities valid for every positive gamma; they do not replace the
model's fixed coefficient in its finite prediction.
"""

from dataclasses import fields
from fractions import Fraction as Q
import hashlib
from math import factorial
from pathlib import Path

import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.mathematics._exact_linear_algebra import (
    exact_matrix_inverse,
    exact_square_matrix_power,
)
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sqrt
from tnfr.physics._organized_loaded_pulse import bound_loaded_c5_pulse
from tnfr.physics._organized_window_observation import bound_loaded_window_observation
from tnfr.utils.io import json_loads


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
            "docs/assets/organized_electrical_feasibility/method-sources-v1.json",
        )
    )
    before = tuple(hashlib.sha256(path.read_bytes()).digest() for path in paths)
    with forbid_organized_response_evaluation():
        yield
    assert tuple(hashlib.sha256(path.read_bytes()).digest() for path in paths) == before


_EDGES = tuple((i, (i + 1) % 5) for i in range(5))
_PORT = (Q(1), Q(-1), Q(0), Q(0), Q(0))


def _dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def _action(matrix, vector):
    return tuple(_dot(row, vector) for row in matrix)


def _laplacian(edges, weight=Q(1)):
    matrix = [[Q(0)] * 5 for _ in range(5)]
    for i, j in edges:
        matrix[i][i] += weight
        matrix[j][j] += weight
        matrix[i][j] -= weight
        matrix[j][i] -= weight
    return tuple(map(tuple, matrix))


@pytest.fixture(scope="module")
def cycle():
    return _laplacian(_EDGES)


def _matrices(cycle, response):
    square = exact_square_matrix_power(cycle, 2)
    metric = tuple(
        tuple(Q(i == j) + response * square[i][j] for j in range(5)) for i in range(5)
    )
    return metric, exact_matrix_inverse(metric)


@pytest.fixture(scope="module")
def window_bound():
    return bound_loaded_window_observation()


@pytest.fixture
def pulse_parameters():
    return {
        "response_lower": Q(1, 100),
        "response_upper": Q(1, 10),
        "amplitude": Q(1, 16),
        "horizon": Q(1, 1024),
        "preparation_radius": Q(1, 2**26),
        "sensor_preparation_radius": Q(1, 2**26),
        "sensor_coupling": Q(1, 64),
    }


def test_shared_pulse_retains_independent_sensor_energy_and_forcing_bound(
    pulse_parameters, window_bound
):
    # At zero form/phase preparation, all initial storage is in the loaded
    # sensor. This checks its factor and held time constant independently.
    parameters = pulse_parameters | {"preparation_radius": Q(0)}
    pulse = bound_loaded_c5_pulse(**parameters)
    k, duration = parameters["sensor_coupling"], parameters["horizon"]
    z0, amplitude = parameters["sensor_preparation_radius"], parameters["amplitude"]
    sensor_energy = k * duration * z0**2 / 2
    assert pulse.initial_energy_upper == sensor_energy > 0
    forcing = tuple(amplitude * v for v in _PORT)
    maximum_growth_rate = _dot(forcing, forcing)
    for gradient in (
        tuple(2 * f for f in forcing),
        tuple(Q(value, 17) for value in (3, -2, 1, -5, 3)),
    ):
        work_minus_form_loss = _dot(gradient, forcing) - _dot(gradient, gradient) / 4
        residual = tuple(q - 2 * f for q, f in zip(gradient, forcing))
        assert (
            work_minus_form_loss == maximum_growth_rate - _dot(residual, residual) / 4
        )
        assert work_minus_form_loss <= maximum_growth_rate
    assert pulse.energy_upper == sensor_energy + duration * maximum_growth_rate
    assert pulse.identity_certified
    # Both consumers must retain the common law's energy and identity fields.
    pulse = bound_loaded_c5_pulse(**pulse_parameters)
    for name in ("initial_energy_upper", "energy_upper", "identity_certified"):
        assert getattr(pulse, name) == getattr(window_bound, name)


def test_shared_pulse_admits_long_energy_windows_without_observation_denominator(
    pulse_parameters,
):
    parameters = pulse_parameters | {
        "horizon": Q(16),
        "amplitude": Q(1, 1024),
        "preparation_radius": Q(0),
        "sensor_preparation_radius": Q(0),
    }
    pulse = bound_loaded_c5_pulse(**parameters)
    assert pulse.initial_energy_upper == 0
    assert pulse.energy_upper == Q(1, 32768) < pulse.storage_barrier
    assert pulse.identity_certified
    with pytest.raises(ValueError, match="horizon and sensor coupling"):
        bound_loaded_window_observation(**parameters)
    energetic = bound_loaded_c5_pulse(**(parameters | {"amplitude": Q(1)}))
    assert energetic.energy_upper == 32 and not energetic.identity_certified


@pytest.mark.parametrize(
    "parameter",
    (
        "response_lower",
        "response_upper",
        "amplitude",
        "horizon",
        "preparation_radius",
        "sensor_preparation_radius",
        "sensor_coupling",
    ),
)
def test_shared_pulse_requires_and_readmits_every_primitive(
    parameter, pulse_parameters
):
    missing = pulse_parameters.copy()
    del missing[parameter]
    with pytest.raises(TypeError, match=parameter):
        bound_loaded_c5_pulse(**missing)
    for invalid in (True, float("nan"), float("inf"), "1"):
        with pytest.raises((TypeError, ValueError)):
            bound_loaded_c5_pulse(**(pulse_parameters | {parameter: invalid}))


@pytest.mark.parametrize(
    "overrides",
    (
        {"response_lower": 0},
        {"response_lower": Q(1, 5)},
        {"response_upper": -1},
        {"amplitude": 0},
        {"horizon": 0},
        {"preparation_radius": -1},
        {"sensor_preparation_radius": -1},
        {"sensor_coupling": 0},
        {"sensor_coupling": -1},
    ),
)
def test_shared_pulse_rejects_invalid_energy_domains(overrides, pulse_parameters):
    with pytest.raises(ValueError):
        bound_loaded_c5_pulse(**(pulse_parameters | overrides))


def test_shared_pulse_normalizes_large_fraction_payloads_before_energy_arithmetic(
    pulse_parameters,
):
    np = pytest.importorskip("numpy")
    parameters = pulse_parameters | {
        "amplitude": Q(np.int64(2**32), np.int64(3)),
        "preparation_radius": Q(0),
        "sensor_preparation_radius": Q(0),
    }
    result = bound_loaded_c5_pulse(**parameters)
    assert result.initial_energy_upper == 0
    assert result.energy_upper == Q(2**55, 9)
    assert not result.identity_certified
    for field in fields(result):
        value = getattr(result, field.name)
        if isinstance(value, Q):
            assert type(value.numerator) is type(value.denominator) is int
    with pytest.raises((TypeError, ValueError)):
        bound_loaded_c5_pulse(
            **(pulse_parameters | {"sensor_coupling": np.bool_(True)})
        )


@pytest.mark.parametrize("response", (Q(1, 100), Q(1, 50), Q(1, 10), Q(2)))
def test_voltage_port_gain_uses_actual_matrix_input(cycle, response):
    bound = bound_loaded_window_observation(
        response_lower=response, response_upper=response
    )
    _, inverse = _matrices(cycle, response)
    gain = _dot(_PORT, _action(cycle, _action(inverse, _PORT)))
    assert gain == bound.response_gain_lower == bound.response_gain_upper
    # The physical voltage port differs from the earlier form contrast.
    assert gain != _dot(_PORT, _action(inverse, _PORT))
    assert 0 < gain < _dot(_PORT, _action(cycle, _PORT)) == 6
    port_gradient = _action(cycle, _PORT)
    assert _dot(port_gradient, port_gradient) == 20 < 25


@pytest.mark.parametrize("response", (Q(0), Q(1, 100), Q(1, 10)))
def test_loaded_storage_retains_sensor_state_and_matrix_backreaction(cycle, response):
    metric, inverse = _matrices(cycle, response)
    form = tuple(Q(value, 17) for value in (3, -2, 1, -5, 3))
    gamma, coupling, sensor_time, sensor = Q(3, 7), Q(1, 64), Q(1, 1024), Q(2, 19)
    forcing = tuple(Q(1, 16) * value for value in _PORT)
    # Exact sine currents from independently summed quarter-turn edge phases.
    phases, sine_values = (0, 1, -1, 2, 0), (Q(0), Q(1), Q(0), Q(-1))
    source = [Q(0)] * 5
    for i, j in _EDGES:
        current = sine_values[(phases[j] - phases[i]) % 4]
        source[i] += current
        source[j] -= current
    gradient = _action(cycle, form)
    port = _dot(_PORT, gradient)
    sensor_rate = (port - sensor) / sensor_time
    internal = tuple((-q + gamma * s) / 4 for q, s in zip(gradient, source))
    applied = tuple(f - coupling * v * (port - sensor) for f, v in zip(forcing, _PORT))
    form_rate = _action(inverse, tuple(a + b for a, b in zip(internal, applied)))
    phase_rate = tuple(gamma * q / 4 for q in gradient)
    form_gradient = _action(metric, gradient)
    object_rate = _dot(form_gradient, form_rate) - _dot(source, phase_rate)
    sensor_storage_rate = coupling * sensor_time * sensor * sensor_rate
    work = _dot(gradient, forcing)
    object_loss = _dot(gradient, gradient) / 4
    sensor_loss = coupling * (port - sensor) ** 2
    assert sensor_storage_rate != 0 and sensor_loss > 0
    assert object_rate + sensor_storage_rate == work - object_loss - sensor_loss
    assert object_rate != work - object_loss - sensor_loss
    assert sum(form_rate) == sum(phase_rate) == 0
    phase_blind_rate = _action(
        inverse, tuple(-q / 4 + f for q, f in zip(gradient, applied))
    )
    assert _dot(form_gradient, phase_blind_rate) + sensor_storage_rate == (
        work - object_loss - sensor_loss
    )
    if response:
        wrong_rate = tuple(a + b for a, b in zip(_action(inverse, internal), applied))
        assert wrong_rate != form_rate
        wrong_balance = (
            _dot(form_gradient, wrong_rate)
            - _dot(source, phase_rate)
            + sensor_storage_rate
        )
        assert wrong_balance != work - object_loss - sensor_loss


def _exp_minus_one_bounds():
    # The alternating Taylor series has decreasing terms; odd/even truncations
    # bound exp(-1) independently of the producer's positive exponential bound.
    upper = sum((Q((-1) ** n, factorial(n)) for n in range(25)), Q(0))
    lower = upper - Q(1, factorial(25))
    return lower, upper


def test_finite_average_retains_sensor_memory_and_differs_from_endpoint(window_bound):
    lower_exp, upper_exp = _exp_minus_one_bounds()
    lower, upper = window_bound.average_ramp_factor_bounds
    # Solve z_x+z=x by series: z=sum(n>=2)(-1)^n*x^n/n!, then integrate.
    # This is the sensor's exact response to a prescribed ramp, not a simulated
    # loaded trajectory or an assumption that actual U follows that ramp.
    average_upper = sum((Q((-1) ** n, factorial(n + 1)) for n in range(2, 25)), Q(0))
    average_lower = average_upper - Q(1, factorial(26))
    assert average_lower <= lower <= upper <= average_upper
    assert Q(1, 2) - upper_exp <= lower <= upper <= Q(1, 2) - lower_exp
    assert Q(1, 8) < lower < upper < Q(1, 7)
    assert upper < lower_exp  # Endpoint ramp response is exp(-1), not its average.
    assert 1 - upper_exp > Q(3, 5)  # Initial sensor state survives averaging.

    # For y=c0+c1*x+c2*x², z=P(x)+C*exp(-x). Verify the polynomial part of
    # z_x+z=y coefficient by coefficient, retaining C=z(0)-P(0).
    c0, c1, c2, initial = Q(2, 7), Q(-3, 11), Q(5, 13), Q(1, 17)
    p2, p1, p0 = c2, c1 - 2 * c2, c0 - c1 + 2 * c2
    memory = initial - p0
    assert (p0 + p1, p1 + 2 * p2, p2) == (c0, c1, c2)
    assert p0 + memory == initial and memory != 0
    average = I(p0 + p1 / 2 + p2 / 3) + memory * I(1 - upper_exp, 1 - lower_exp)
    memory_free = I(p0 + p1 / 2 + p2 / 3)
    assert average.hi < memory_free.lo


def test_declared_bounds_separate_matrix_loading_and_preserve_identity(window_bound):
    bound = window_bound
    assert bound.identity_certified and bound.response_separated
    assert bound.energy_upper < Q(1, 2**16) < bound.storage_barrier
    assert bound.response_gap > Q(1, 2**19)
    assert bound.positive_observation_bounds[1] < bound.control_observation_bounds[0]
    assert bound.response_gap == (
        bound.control_observation_bounds[0] - bound.positive_observation_bounds[1]
    )
    assert bound.state_norm_upper < Q(1, 8000)
    assert 0 < bound.pulse_absolute_work_upper < Q(1, 10**6)
    assert 0 < bound.pulse_loss_upper < Q(1, 10**8)
    assert 0 < bound.sensor_loss_upper < Q(1, 10**8)
    # The common coarse alpha>1/8 also discriminates the complete r interval.
    coarse_gap = bound.amplitude * bound.horizon * (6 - bound.response_gain_upper) / 8
    coarse_gap -= 2 * (bound.response_remainder + bound.observation_error)
    assert Q(1, 2**21) < coarse_gap < bound.response_gap


def test_phase_blind_reference_has_a_constructive_permitted_noise_collision(
    window_bound,
):
    bound = window_bound
    # Separate triangle/convolution majorant for the actual loaded form,
    # using gamma<1/3000 rather than replacing gamma in either evolution law.
    delta, duration, gain, coupling = (
        bound.preparation_radius,
        bound.horizon,
        Q(1, 3000),
        bound.sensor_coupling,
    )
    form_upper = (
        delta
        + duration
        * (
            2 * bound.amplitude
            + gain * delta
            + 2 * coupling * bound.sensor_preparation_radius
        )
    ) / (1 - duration * (1 + 20 * coupling + gain**2 * duration))
    assert form_upper < Q(1, 8000)
    independent = (
        5
        * gain
        * (delta * duration + gain * Q(1, 4000) * duration**2 / 2)
        / (1 - (1 + 20 * coupling) * duration)
    )
    assert independent < Q(1, 2**44) < bound.observation_error / 100
    assert bound.phase_blind_difference_upper <= independent
    assert bound.phase_blind_difference_upper >= (
        5 * gain * delta * duration / (1 - (1 + 20 * coupling) * duration)
    )
    assert bound.phase_blind_difference_upper < bound.observation_error / 100
    assert bound.phase_blind_records_overlap
    # For every |d|<=B, errors +/-d/2 put both outputs at their midpoint.
    # Endpoint witnesses exercise the full symmetric discrepancy enclosure.
    center = Q(3, 11)
    for difference in (
        -bound.phase_blind_difference_upper,
        bound.phase_blind_difference_upper,
    ):
        full, linear = center, center + difference
        observed = (full + linear) / 2
        assert abs(observed - full) <= bound.observation_error
        assert abs(observed - linear) <= bound.observation_error
    exact_readout = bound_loaded_window_observation(observation_error=0)
    assert not exact_readout.phase_blind_records_overlap
    assert exact_readout.phase_blind_difference_upper > 0


def test_energy_and_winding_do_not_admit_the_local_preparation(window_bound):
    kappa, rho = 2 * pi_interval() / 5, window_bound.identity_radius
    boundary_gaps = (kappa + rho,) + (kappa - rho / 4,) * 4
    local_potential = sum((1 - cos(gap) for gap in boundary_gaps), I(0))
    twist_potential = 5 * (1 - cos(kappa))
    assert (local_potential - twist_potential).lo >= rho**2 / 8 > 0
    # At the limiting gap tuple (pi,pi/4,pi/4,pi/4,pi/4), exact radicals
    # give a strict negative sign. Continuity moves inside (-pi,pi), then
    # IVT gives a nonlocal equal-energy state without any root search.
    assert Q(7, 5) ** 2 < 2 and Q(9, 4) ** 2 > 5
    endpoint = 6 - 2 * sqrt(I(2))
    exact_twist = (25 - 5 * sqrt(I(5))) / 4
    assert exact_twist.lo <= twist_potential.hi
    assert twist_potential.lo <= exact_twist.hi
    assert endpoint.hi < Q(16, 5) < Q(55, 16) < exact_twist.lo
    assert (endpoint - exact_twist).hi < Q(-19, 80)
    assert kappa.hi + rho < pi_interval().lo
    # Every root beyond the local boundary has a gap deviation >rho.
    # Cauchy-Schwarz excludes the full delta preparation independently of
    # the equality of its voltage, winding and total storage observations.
    assert 2 * window_bound.preparation_radius**2 < rho**2


@pytest.mark.parametrize("response", (Q(1, 100), Q(1, 10)))
def test_physical_sensor_rows_storage_and_clock_product(cycle, response, window_bound):
    centered_inverse = _laplacian(((i, (i + 2) % 5) for i in range(5)), Q(1, 5))
    form = tuple(Q(value, 17) for value in (3, -2, 1, -5, 3))
    form_rate = tuple(Q(value, 19) for value in (-2, 5, 1, -3, -1))
    a0, gamma, coupling, duration, sensor = (
        Q(3, 7),
        Q(1, 37),
        window_bound.sensor_coupling,
        window_bound.horizon,
        Q(2, 13),
    )
    for capacitance, conductance in ((Q(5, 11), Q(7, 17)), (Q(13, 19), Q(2, 3))):
        b = 4 * conductance / capacitance
        v0 = a0 * gamma * conductance / capacitance
        junction = a0 * gamma**2 * conductance**2 / capacitance
        voltage = tuple(v0 * q for q in _action(cycle, form))
        voltage_rate = tuple(b * v0 * q for q in _action(cycle, form_rate))
        physical_matrix = tuple(
            tuple(capacitance * (lplus + response * l) for lplus, l in zip(row, edge))
            for row, edge in zip(centered_inverse, cycle)
        )
        metric, _ = _matrices(cycle, response)
        assert _action(physical_matrix, voltage_rate) == tuple(
            capacitance * b * v0 * q for q in _action(metric, form_rate)
        )
        port = _dot(_PORT, _action(cycle, form))
        sensor_rate = (port - sensor) / duration
        gs, cs = 4 * conductance * coupling, capacitance * coupling * duration
        physical_current = gs * (_dot(_PORT, voltage) - v0 * sensor)
        assert cs * b * v0 * sensor_rate == physical_current
        assert physical_current / (capacitance * b * v0) == coupling * (port - sensor)
        assert cs * (v0 * sensor) ** 2 / 2 == (
            a0 * junction * coupling * duration * sensor**2 / 2
        )
        assert gs * (_dot(_PORT, voltage) - v0 * sensor) ** 2 / b == (
            a0 * junction * coupling * (port - sensor) ** 2
        )
        form_storage = _dot(form, _action(cycle, _action(metric, form))) / 2
        assert _dot(voltage, _action(physical_matrix, voltage)) / 2 == (
            a0 * junction * form_storage
        )
        allowed_voltage_error = v0 * window_bound.observation_error
        laboratory_duration = duration / b
        assert allowed_voltage_error * laboratory_duration == (
            a0 * gamma * duration * window_bound.observation_error / 4
        )


def test_failed_sufficient_bounds_are_reported_without_claiming_trajectory_failure():
    noisy = bound_loaded_window_observation(observation_error=1)
    assert noisy.identity_certified and not noisy.response_separated
    energetic = bound_loaded_window_observation(amplitude=1)
    assert not energetic.identity_certified
    assert energetic.energy_upper > energetic.storage_barrier
    broad = bound_loaded_window_observation(preparation_radius=Q(1, 20))
    assert not broad.identity_certified and not broad.response_separated
    assert broad.pulse_absolute_work_upper > 0


@pytest.mark.parametrize(
    "parameter",
    (
        "response_lower",
        "response_upper",
        "amplitude",
        "horizon",
        "preparation_radius",
        "sensor_preparation_radius",
        "sensor_coupling",
        "observation_error",
    ),
)
@pytest.mark.parametrize("invalid", (True, float("nan"), float("inf"), "1"))
def test_admission_precedes_every_budget_calculation(parameter, invalid):
    with pytest.raises((TypeError, ValueError)):
        bound_loaded_window_observation(**{parameter: invalid})


@pytest.mark.parametrize(
    "overrides",
    (
        {"response_lower": 0},
        {"response_lower": Q(1, 5)},
        {"response_upper": -1},
        {"amplitude": 0},
        {"horizon": 0},
        {"horizon": 1},
        {"horizon": Q(16, 37)},
        {"preparation_radius": -1},
        {"sensor_preparation_radius": -1},
        {"sensor_coupling": -1},
        {"sensor_coupling": 0},
        {"observation_error": -1},
    ),
)
def test_invalid_mathematical_domains_reject(overrides):
    with pytest.raises(ValueError):
        bound_loaded_window_observation(**overrides)


def test_exact_admission_normalizes_fixed_width_fraction_payloads(window_bound):
    np = pytest.importorskip("numpy")
    payload = Q(np.int64(1), np.int64(2**26))
    assert (
        bound_loaded_window_observation(sensor_preparation_radius=payload)
        == window_bound
    )
    assert (
        bound_loaded_window_observation(preparation_radius=float(payload))
        == window_bound
    )
    amplitude = Q(np.int64(2**32), np.int64(3))
    actual = bound_loaded_window_observation(amplitude=amplitude)
    assert actual == bound_loaded_window_observation(amplitude=Q(2**32, 3))

    def normalized(value):
        if isinstance(value, Q):
            assert type(value.numerator) is type(value.denominator) is int
        elif isinstance(value, tuple):
            for item in value:
                normalized(item)

    for field in fields(actual):
        normalized(getattr(actual, field.name))
    with pytest.raises((TypeError, ValueError)):
        bound_loaded_window_observation(sensor_coupling=np.bool_(True))
    zero = bound_loaded_window_observation(
        preparation_radius=0, sensor_preparation_radius=0, observation_error=0
    )
    assert zero.identity_certified and zero.response_separated
    assert zero.response_gap > window_bound.response_gap


def test_primary_method_receipt_preserves_no_experiment_scope():
    root = Path(__file__).resolve().parents[2]
    path = root / "docs/assets/organized_finite_window_design/method-sources-v1.json"
    receipt = json_loads(path.read_bytes())
    assert receipt["schema"] == "tnfr.organized-finite-window-design.method-sources.v1"
    for flag in (
        "physical_dataset_selected",
        "measured_arrays_downloaded",
        "response_fit_performed",
        "reserved_response_evaluated",
        "physical_apparatus_admitted",
    ):
        assert receipt[flag] is False
    assert (root / receipt["owner"]).is_file()
    assert receipt["sources"]
    for source in receipt["sources"]:
        assert source["canonical_url"].startswith("https://")
        assert source["reviewed_sections"] and source["inference_limits"]
        assert source["review_method"] == "web_source_text"
        assert source["retrieval_status"] == "web_review_only"
        assert source["sha256"] is None and source["bytes"] is None
        assert source["optional_local_cache_path"] is None
