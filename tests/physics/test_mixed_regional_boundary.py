"""Independent finite regional-boundary controls for the acquired composite.

Generic field points below are not reconstructed acquired endpoints. The two
admitted histories share a pre-probe global mean, not identical fine phases.
No trajectory, attained endpoint, nominal reset or source replay is supplied.
"""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as Q
from itertools import repeat
from math import factorial

import mpmath
import numpy as np
import pytest

from tests.sine_evidence_helpers import forbid_organized_response_evaluation
from tnfr.dynamics import integrators, runtime
from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.operators import _coupling_stage_kernel, network_stage, word_execution
from tnfr.operators.definitions_base import Operator
from tnfr.physics import _mixed_regional_boundary as owner
from tnfr.physics import _sine_flow as flow_owner

ERROR = Q(1, 10**10)
ETA = Q(2, 10**69)
DELTA = Q(1, 1024)
RADIUS = Q(1, 32)
TIME = Q(1, 32)
AMPLITUDE = Q(1, 16)
NORM = Q(9, 512)
MEAN = NORM / 3
RESIDUAL = Q(1, 10**5)
NOISE = Q(1, 10**6)
ARITHMETIC = Q(1, 10**35)


def _mp(mp, value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(mp, interval, value):
    assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
    assert interval.width < Q(1, 10**30)


def _turns():
    return tuple(Q((1 + i // 9) * (i % 9 - 4), 9) for i in range(18))


def _support():
    bins = tuple(int((9 * value) % 9) for value in _turns())
    pairs = {
        (i, j)
        for i in range(18)
        for j in range(i + 1, 18)
        if (
            min((bins[j] - bins[i]) % 9, (bins[i] - bins[j]) % 9) in (1, 2)
            or (i // 9 != j // 9 and bins[i] == bins[j])
        )
    }
    old = tuple(
        sorted(
            tuple(sorted((9 * r + j, 9 * r + (j + 1) % 9)))
            for r in (0, 1)
            for j in range(9)
        )
    )
    return old, tuple(sorted(pairs - set(old)))


def _weights():
    return tuple(Q(3, 4) + Q(i % 5, 16) for i in range(63))


def _edges(weights=None):
    old, new = _support()
    weights = _weights() if weights is None else weights
    return tuple((a, b, Q(1)) for a, b in old) + tuple(
        (a, b, w) for (a, b), w in zip(new, weights, strict=True)
    )


def _state(forms, errors):
    return tuple(I(x) for x in forms) + tuple(
        2 * pi_interval() * turn + error
        for turn, error in zip(_turns(), errors, strict=True)
    )


def _oracle(mp, forms, errors, weights=None):
    x = tuple(_mp(mp, value) for value in forms)
    theta = tuple(
        2 * mp.pi * _mp(mp, turn) + _mp(mp, error)
        for turn, error in zip(_turns(), errors, strict=True)
    )
    q, sine = [mp.mpf(0)] * 18, [mp.mpf(0)] * 18
    for a, b, weight in _edges(weights):
        gradient = _mp(mp, weight) * (x[a] - x[b])
        current = mp.sin(theta[b] - theta[a])
        q[a] += gradient
        q[b] -= gradient
        sine[a] += current
        sine[b] -= current
    gamma = 1 / (1023 * mp.pi)
    dx = tuple((-a + gamma * b) / 9 for a, b in zip(q, sine))
    phase = tuple(gamma * value / 9 for value in q)
    return x, tuple(q), tuple(sine), dx, phase


def _assess(**changes):
    arguments = dict(
        form_error_bound=ERROR,
        phase_error_bound=ERROR,
        starting_gap_upper_bound=DELTA / 8,
        readout_error_bound=NOISE,
    )
    arguments.update(changes)
    return owner._assess_mixed_regional_boundary(**arguments)


def _predict(**changes):
    arguments = dict(
        regional_mean_reading=MEAN,
        regional_mean_error_bound=NOISE,
        readout_error_bound=NOISE,
    )
    arguments.update(changes)
    return owner._predict_mixed_regional_transfer(**arguments)


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_event_or_prior_producer():
    static_factory = flow_owner._sine_rate_evaluator

    def forbidden(*args, **kwargs):
        pytest.fail("finite boundary controls cannot advance or replay a source")

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
        with pytest.MonkeyPatch.context() as static:
            static.setattr(flow_owner, "_sine_rate_evaluator", static_factory)
            yield


@pytest.fixture(scope="module")
def mp():
    context = mpmath.mp.clone()
    context.dps = 200
    return context


@pytest.fixture(scope="module")
def report():
    return _assess()


@pytest.fixture(scope="module")
def fields():
    return {
        "charge": owner._mixed_regional_boundary_field(
            new_edge_conductances=_weights(), stage="charge", charge_amplitude=AMPLITUDE
        ),
        "probe": owner._mixed_regional_boundary_field(
            new_edge_conductances=_weights(), stage="probe"
        ),
    }


def _product(left, right):
    return tuple(
        tuple(sum(a * b for a, b in zip(row, column)) for column in zip(*right))
        for row in left
    )


def _exponential_series(matrix, time):
    """Signed 3x3 Taylor polynomial, with an independent absolute norm tail."""
    size = len(matrix)
    term = tuple(tuple(Q(int(i == j)) for j in range(size)) for i in range(size))
    result = term
    scaled = tuple(tuple(time * value for value in row) for row in matrix)
    norm = max(sum(abs(value) for value in row) for row in scaled)
    for n in range(1, 61):
        term = tuple(
            tuple(value / n for value in row) for row in _product(term, scaled)
        )
        result = tuple(
            tuple(a + b for a, b in zip(row, other)) for row, other in zip(result, term)
        )
    tail = norm**61 / factorial(61) / (1 - norm / 62)
    return result, tail


def _nominal_weights(mp):
    phase = tuple(
        2 * mp.pi * _mp(mp, value) + (-1 if i < 9 else 1) * _mp(mp, DELTA) / 2
        for i, value in enumerate(_turns())
    )
    return tuple(
        1 - abs((phase[b] - phase[a] + mp.pi) % (2 * mp.pi) - mp.pi) / (2 * mp.pi)
        for a, b in _support()[1]
    )


def test_complete_regional_rows_and_global_balance_are_independent(mp, fields):
    forms = tuple(Q((i % 7) - 3, 4096) for i in range(18))
    errors = tuple(Q((i % 5) - 2, 4096) for i in range(18))
    state = _state(forms, errors)
    x, q, sine, intrinsic, phase = _oracle(mp, forms, errors)
    mean_phase_error = sum(errors) / 18
    y = tuple(_mp(mp, value - mean_phase_error) for value in errors)
    norm_squared = sum(value**2 for value in x + y)
    assert len(_edges()) == 81
    assert sum(a < 9 <= b for a, b, _ in _edges()) == 45
    assert any(value != 0 for value in phase)
    for stage, field in fields.items():
        inputs = tuple(
            (
                _mp(mp, AMPLITUDE) * (1 if i < 9 else -1)
                if stage == "charge"
                else (-x[i] if i < 9 else mp.mpf(0))
            )
            for i in range(18)
        )
        expected = tuple(a + b for a, b in zip(intrinsic, inputs)) + phase
        actual = field(state)
        assert len(actual) == 36
        for interval, value in zip(actual, expected, strict=True):
            _contains(mp, interval, value)
        _contains(mp, sum(actual[:18], I(0)), sum(inputs))
        _contains(mp, sum(actual[18:], I(0)), mp.mpf(0))
        if stage == "probe":
            _contains(mp, sum(actual[:18], I(0)), -sum(x[:9]))
        norm_derivative = sum(a * b for a, b in zip(x, expected[:18]))
        norm_derivative += sum(a * b for a, b in zip(y, phase))
        bound = 2 * norm_squared / 3000
        if stage == "charge":
            bound += mp.sqrt(18 * norm_squared) * _mp(mp, AMPLITUDE)
        assert norm_derivative <= bound
    assert fields["charge"](state)[18:] == fields["probe"](state)[18:]
    # Thus pre-probe global mean stays zero. During the probe its change
    # instead reconstructs Q_A and cannot be given to the weaker predictor.
    assert abs(sum(q)) < mp.mpf("1e-190")
    assert abs(sum(sine)) < mp.mpf("1e-190")


def test_nominal_equitable_projection_retains_actual_affinity_association(mp):
    weights = _nominal_weights(mp)
    cross_strengths = [mp.mpf(0)] * 18
    q = [mp.mpf(0)] * 18
    forms = tuple(_mp(mp, Q((7 * i) % 13 - 6, 2048)) for i in range(18))
    for (a, b), weight in zip(_support()[1], weights):
        if a < 9 <= b:
            cross_strengths[a] += weight
            cross_strengths[b] += weight
            current = weight * (forms[a] - forms[b])
            q[a] += current
            q[b] -= current
    cross = mp.mpf(13) / 3 - _mp(mp, DELTA) / (2 * mp.pi)
    assert all(abs(value - cross) < mp.mpf("1e-190") for value in cross_strengths)
    mean_a, mean_b = sum(forms[:9]) / 9, sum(forms[9:]) / 9
    assert abs(sum(q[:9]) / 81 - cross / 9 * (mean_a - mean_b)) < mp.mpf("1e-190")
    assert abs(sum(q)) < mp.mpf("1e-190")
    # The nominal reference is used through its equitable sums, not by
    # replacing every actual source state with a two-constant carrier.


def test_non_equitable_family_is_not_misidentified_as_an_actual_source(mp):
    old, new = _support()
    marked = next(i for i, (a, b) in enumerate(new) if a == 0 and b >= 9)
    weights = tuple(Q(3, 4) if i == marked else Q(1) for i in range(len(new)))
    forms = (Q(1, 1024), -Q(1, 1024)) + (Q(0),) * 16
    assert sum(forms[:9]) == sum(forms[9:]) == 0
    x, q, sine, dx, _ = _oracle(mp, forms, (Q(0),) * 18, weights)
    assert abs(sum(q[:9]) + _mp(mp, Q(1, 4096))) < mp.mpf("1e-190")
    assert abs(sum(dx[:9]) / 9 - _mp(mp, Q(1, 331776))) < mp.mpf("1e-190")
    assert max(abs(value) for value in sine) < mp.mpf("1e-190")
    field = owner._mixed_regional_boundary_field(
        new_edge_conductances=weights, stage="charge"
    )
    _contains(
        mp, sum(field(_state(forms, (Q(0),) * 18))[:9], I(0)) / 9, sum(dx[:9]) / 9
    )
    assert x != (mp.mpf(0),) * 18
    # This legal broad-family coefficient point is not asserted to be an
    # attained proposal. The actual image has a much smaller weight residual.


def test_regional_work_can_raise_relative_storage_and_counts_absolute_input(mp):
    weights = _nominal_weights(mp)
    x = (mp.mpf(1) / 512,) * 9 + (mp.mpf(3) / 1024,) * 9
    q = [mp.mpf(0)] * 18
    for (a, b), weight in zip(_support()[1], weights):
        current = weight * (x[a] - x[b])
        q[a] += current
        q[b] -= current
    loss = -sum(value**2 for value in q) / 9
    charge_work = _mp(mp, AMPLITUDE) * (sum(q[:9]) - sum(q[9:]))
    probe_work = -sum(a * b for a, b in zip(q[:9], x[:9]))
    assert charge_work < 0 < -charge_work
    assert probe_work > 0 and loss + probe_work > 0
    # Nevertheless the full form norm sees the regional leak as dissipative.
    norm_derivative = -sum(a * b for a, b in zip(x, q)) / 9 - sum(
        value**2 for value in x[:9]
    )
    assert norm_derivative < 0
    assert 9 * AMPLITUDE * TIME - 9 * AMPLITUDE * TIME == 0
    assert 18 * AMPLITUDE * TIME == Q(9, 256)
    # Aligned generic state, not a claimed attained endpoint. This full-row
    # storage calculation excludes inherited passivity from the uniform port.


def test_jets_keep_both_phase_rows_and_region_addressed_input(mp, fields):
    forms = tuple(Q((i % 7) - 3, 4096) for i in range(18))
    errors = tuple(Q((i % 5) - 2, 4096) for i in range(18))
    direction = tuple(Q(1) if i < 9 else -Q(1) for i in range(18))
    state = _state(forms, errors)
    jets = tuple(
        Jet((value, I(d))) for value, d in zip(state, direction + (Q(0),) * 18)
    )
    x, _, _, intrinsic, phase = _oracle(mp, forms, errors)
    gradient = [Q(0)] * 18
    for a, b, weight in _edges():
        current = weight * (direction[a] - direction[b])
        gradient[a] += current
        gradient[b] -= current
    gamma = 1 / (1023 * mp.pi)
    assert any(value for value in gradient)
    for stage, field in fields.items():
        inputs = tuple(
            (
                _mp(mp, AMPLITUDE * direction[i])
                if stage == "charge"
                else (-x[i] if i < 9 else mp.mpf(0))
            )
            for i in range(18)
        )
        constants = tuple(a + b for a, b in zip(intrinsic, inputs)) + phase
        derivatives = tuple(
            -_mp(mp, value) / 9
            - (int(stage == "probe" and i < 9) * _mp(mp, direction[i]))
            for i, value in enumerate(gradient)
        )
        derivatives += tuple(gamma * _mp(mp, value) / 9 for value in gradient)
        for jet, constant, derivative in zip(field(jets), constants, derivatives):
            _contains(mp, jet.coeffs[0], constant)
            _contains(mp, jet.coeffs[1], derivative)


def test_projected_residual_prices_weight_sine_and_coefficient_channels(mp, report):
    # An exact generic perturbation checks the transport identity without
    # claiming that this convenient state is an acquired endpoint.
    old, new = _support()
    base = tuple(Q(4, 5) if a < 9 <= b else Q(1) for a, b in new)
    marked = next(i for i, (a, b) in enumerate(new) if a == 0 and b >= 9)
    deviation = 10**9 * ETA
    changed = tuple(
        value + (deviation if i == marked else 0) for i, value in enumerate(base)
    )
    forms = (NORM / 4, -NORM / 4) + (Q(0),) * 16
    flux = Q(0)
    baseline_flux = Q(0)
    for (a, b), w, w0 in zip(new, changed, base, strict=True):
        if a < 9 <= b:
            flux += w * (forms[a] - forms[b])
            baseline_flux += w0 * (forms[a] - forms[b])
    assert baseline_flux == 0
    transport_residual = (flux - baseline_flux) / 81
    assert transport_residual == deviation * NORM / 324 > 0
    assert transport_residual < Q(10, 9) * NORM * deviation
    errors = (Q(1, 4096), -Q(1, 4096)) + (Q(0),) * 16
    _, _, sine, _, phase = _oracle(mp, forms, errors, base)
    phase_residual = sum(sine[:9]) / (81 * 1023 * mp.pi)
    assert abs(phase_residual) < _mp(mp, Q(10, 9) * NORM / 3000)
    assert any(value != 0 for value in phase)
    reference = report.reference
    components = (
        Q(10, 9) * NORM / 3000,
        Q(10, 9) * NORM * deviation,
        Q(2, 3) * NORM * reference.coupling_radius,
    )
    assert all(value > 0 for value in components)
    assert sum(components) == reference.residual_calculated_upper_bound < RESIDUAL
    assert reference.residual_certified
    # Even zero reading noise does not remove this full-state projection cost.
    assert report.model_error_upper_bound == 9 * TIME**2 * RESIDUAL / 2 > 0


def test_reference_gain_has_independent_signed_series_and_closed_oracles(
    mp, report, record_property
):
    reference = report.reference
    actual_k = mp.mpf(13) / 27 - _mp(mp, DELTA) / (18 * mp.pi)
    lo, hi = reference.coupling_bounds
    assert _mp(mp, lo) <= actual_k <= _mp(mp, hi)
    k = reference.reference_coupling
    assert k == (lo + hi) / 2 and reference.coupling_radius == (hi - lo) / 2
    assert 0 < lo <= k <= hi < Q(1, 2)
    matrix = ((-1 - k, k, Q(0)), (k, -k, Q(0)), (Q(9), Q(0), Q(0)))
    assert reference.generator == matrix
    polynomial, tail = _exponential_series(matrix, TIME)
    series_gain = polynomial[2][0] - polynomial[2][1]
    gain = reference.gain_bounds
    assert 0 <= tail < Q(1, 10**110)
    assert gain[0] <= series_gain - 2 * tail <= series_gain + 2 * tail <= gain[1]
    assert gain[1] - gain[0] < Q(1, 10**45)
    # This closed expression uses the rational reference k0. The true
    # coefficient differs and its entire effect has already been priced in F.
    km, time = _mp(mp, k), _mp(mp, TIME)
    root = mp.sqrt(1 + 4 * km**2)
    closed = 18 * mp.exp(-(1 + 2 * km) * time / 2) * mp.sinh(root * time / 2) / root
    assert _mp(mp, gain[0]) <= closed <= _mp(mp, gain[1])
    assert 9 * TIME * (1 - TIME) <= gain[0] <= gain[1] <= 9 * TIME
    assert reference.coefficient_domain_certified and reference.gain_certified
    assert (
        reference.exponential.lower[2][0] - reference.exponential.upper[2][1] == gain[0]
    )
    assert (
        reference.exponential.upper[2][0] - reference.exponential.lower[2][1] == gain[1]
    )
    record_property("reference_gain_bounds", str(gain))
    record_property("projected_residual_allowance", str(RESIDUAL))


def test_full_norm_tube_preserves_registration_with_supplied_work(
    report, record_property
):
    common = report.successor.completion_reports[0]
    assert report.certified and report.status == "certified"
    assert report.source_certified and report.finite_protocol_ready_age_exists
    assert report.conditional_ready_source and report.full_state_tube_certified
    assert common.initial_energy_upper_bound < Q(1, 65536)
    assert (
        report.initial_norm_squared_upper_bound == 5 * common.initial_energy_upper_bound
    )
    assert report.initial_norm_squared_upper_bound < Q(9, 1024) ** 2
    assert (
        report.initial_norm_margin
        == Q(9, 1024) ** 2 - report.initial_norm_squared_upper_bound
        > 0
    )
    assert 18 < Q(17, 4) ** 2
    exponent = 2 * Q(1, 3000) * (TIME + TIME)
    assert exponent == Q(1, 24000)
    assert 1 / (1 - exponent) < Q(36, 35)
    tube = (Q(9, 1024) + Q(17, 4) * AMPLITUDE * TIME) * Q(36, 35)
    assert tube == report.norm_tube_upper_bound == NORM < RADIUS
    assert report.norm_tube_margin == RADIUS - NORM > 0
    assert (
        report.marked_gap_change_upper_bound
        == 4 * Q(1, 3000) * RADIUS * (TIME + TIME)
        == Q(1, 384000)
    )
    assert report.retained_gap_upper_bound == DELTA / 8 + Q(1, 384000) < DELTA / 4
    assert report.registration_margin == DELTA / 4 - report.retained_gap_upper_bound > 0
    assert report.joint_registration_retained_certified
    assert (
        report.charge_storage_work_upper_bound
        == 18 * NORM * Q(17, 4) * AMPLITUDE * TIME
    )
    assert report.probe_storage_work_upper_bound == 18 * NORM**2 * TIME
    assert report.supplied_signed_form_amount == 0
    assert report.supplied_absolute_form_amount_upper_bound == Q(9, 256)
    assert (
        report.controlled_charge_channels,
        report.controlled_probe_channels,
        report.predictor_regional_mean_coordinates,
    ) == (18, 9, 1)
    assert report.regions == (tuple(range(9)), tuple(range(9, 18)))
    assert report.existence_gap_threshold == DELTA / 8
    assert report.preparation_mean_bounds == (-MEAN, MEAN)
    record_property("full_state_norm_tube", str(NORM))
    record_property(
        "retained_registered_gap_bound", str(report.retained_gap_upper_bound)
    )
    record_property("supplied_absolute_form_amount", str(Q(9, 256)))


def test_actual_history_comparison_excludes_preprobe_global_mean_only(
    mp, report, record_property
):
    k = _mp(mp, report.reference.reference_coupling)
    time, amplitude = _mp(mp, TIME), _mp(mp, AMPLITUDE)
    reference_mean_difference = amplitude * (1 - mp.exp(-2 * k * time)) / k
    mean_leading = 2 * AMPLITUDE * TIME * (1 - TIME / 2)
    assert reference_mean_difference >= _mp(mp, mean_leading)
    assert (
        report.charged_regional_mean_gap_lower_bound
        == mean_leading - 2 * TIME * RESIDUAL
        > 0
    )
    leading = 18 * AMPLITUDE * TIME**2 * (1 - TIME / 2) * (1 - TIME)
    assert leading == Q(17577, 16777216) > Q(1, 960)
    loss = (18 * TIME**2 + 9 * TIME**2) * RESIDUAL
    assert loss + 2 * NOISE < Q(3, 10**6)
    assert report.actual_history_transfer_gap_lower_bound == leading - loss
    assert (
        report.recorded_history_transfer_gap_lower_bound
        == leading - loss - 2 * NOISE
        > Q(1, 1000)
    )
    assert (
        report.weak_prediction_error_lower_bound
        == report.recorded_history_transfer_gap_lower_bound / 2
        > Q(2, 10**6)
    )
    assert (
        report.positive_information_certified
        and report.global_history_obstruction_certified
    )
    # The histories originate at the same actual conditional source. Their
    # common input sum is zero before probing, so global mean history is zero.
    # Fine phases need not agree. During probing Q_A=-18*(m_end-m_start), so
    # that later global reading would disclose the reserved output itself.
    record_property(
        "recorded_actual_history_gap_lower_bound",
        str(report.recorded_history_transfer_gap_lower_bound),
    )
    record_property(
        "weak_forecast_error_lower_bound", str(report.weak_prediction_error_lower_bound)
    )
    record_property("negative_shared_observation", "pre_probe_global_mean_history_only")


def test_decoder_signed_noise_and_model_allowance_cover_independent_corners(
    mp, report, record_property
):
    k = _mp(mp, report.reference.reference_coupling)
    root = mp.sqrt(1 + 4 * k**2)
    gain = (
        18
        * mp.exp(-(1 + 2 * k) * _mp(mp, TIME) / 2)
        * mp.sinh(root * _mp(mp, TIME) / 2)
        / root
    )
    model = 9 * TIME**2 * RESIDUAL / 2
    # These are allowance-domain checks, not claims that every corner is an
    # attained nonlinear endpoint or that the residual is independently set.
    for mean in (-MEAN, Q(0), MEAN):
        for reading_error in (-NOISE, NOISE):
            prediction = _predict(regional_mean_reading=mean + reading_error)
            assert prediction.certified and prediction.mean_observation_consistent
            assert prediction.regional_mean_reading_bounds == (
                mean + reading_error - NOISE,
                mean + reading_error + NOISE,
            )
            assert prediction.model_error_upper_bound == model
            assert prediction.mean_error_contribution == 9 * TIME * NOISE
            assert prediction.arithmetic_error_upper_bound <= ARITHMETIC
            assert prediction.record_prediction_error_upper_bound < Q(2, 10**6)
            for residual in (-model, model):
                if mean == 0:
                    # G*0 is exact. Keep rational boundary corners exact
                    # instead of adding separately rounded mpmath values.
                    actual = residual
                    assert (
                        prediction.true_transfer_bounds[0]
                        <= actual
                        <= prediction.true_transfer_bounds[1]
                    )
                    for readout in (-NOISE, NOISE):
                        recorded = actual + readout
                        assert (
                            prediction.recorded_transfer_bounds[0]
                            <= recorded
                            <= prediction.recorded_transfer_bounds[1]
                        )
                        assert (
                            abs(recorded - prediction.forecast)
                            <= prediction.record_prediction_error_upper_bound
                        )
                    continue
                actual = gain * _mp(mp, mean) + _mp(mp, residual)
                assert (
                    _mp(mp, prediction.true_transfer_bounds[0])
                    <= actual
                    <= _mp(mp, prediction.true_transfer_bounds[1])
                )
                for readout in (-NOISE, NOISE):
                    recorded = actual + _mp(mp, readout)
                    assert (
                        _mp(mp, prediction.recorded_transfer_bounds[0])
                        <= recorded
                        <= _mp(mp, prediction.recorded_transfer_bounds[1])
                    )
                    assert abs(recorded - _mp(mp, prediction.forecast)) <= _mp(
                        mp, prediction.record_prediction_error_upper_bound
                    )
    zero = _predict(
        regional_mean_reading=0, regional_mean_error_bound=0, readout_error_bound=0
    )
    assert zero.certified and zero.forecast == 0
    assert zero.true_transfer_bounds == (-model, model)
    assert zero.record_prediction_error_upper_bound == model > 0
    assert (
        report.record_prediction_error_upper_bound
        == 9 * TIME * NOISE
        + model
        + NOISE
        + report.uniform_arithmetic_error_upper_bound
    )
    assert report.record_prediction_error_upper_bound < Q(2, 10**6)
    record_property(
        "regional_forecast_error_upper_bound",
        str(report.record_prediction_error_upper_bound),
    )
    record_property("regional_forecast_target", str(Q(2, 10**6)))


def test_observation_consistency_never_clamps_and_caps_remain_separate():
    touching = _predict(regional_mean_reading=MEAN + NOISE)
    assert touching.certified
    assert touching.regional_mean_reading_bounds == (MEAN, MEAN + 2 * NOISE)
    assert (
        touching.regional_mean_reading_bounds[1] > touching.preparation_mean_bounds[1]
    )
    inconsistent = _predict(regional_mean_reading=MEAN + NOISE + Q(1, 10**100))
    assert not inconsistent.mean_observation_consistent and not inconsistent.certified
    assert (
        "regional_mean_inconsistent_with_preparation"
        in inconsistent.unresolved_conditions
    )
    for key in ("regional_mean_error_bound", "readout_error_bound"):
        outside = _predict(**{key: NOISE + Q(1, 10**100)})
        assert not outside.allowances_admitted and not outside.certified
    gap = _assess(starting_gap_upper_bound=DELTA / 8 + Q(1, 10**100))
    assert gap.source_certified and gap.finite_protocol_ready_age_exists
    assert not gap.starting_gap_domain_admitted and not gap.conditional_ready_source
    noisy = _assess(readout_error_bound=NOISE + Q(1, 10**100))
    assert noisy.joint_registration_retained_certified and not noisy.certified
    zero = _assess(starting_gap_upper_bound=0)
    assert zero.conditional_ready_source and zero.finite_protocol_ready_age_exists
    assert zero.existence_gap_threshold == DELTA / 8 > zero.starting_gap_upper_bound
    # Existence of a fixed positive-threshold ready age never authenticates
    # finite attainment of this independently supplied zero readiness premise.


def test_source_readmission_rebuilds_only_the_relevant_connected_degree_claim(
    monkeypatch, report
):
    source = report.successor
    common, split = source.completion_reports
    unrelated = replace(
        source,
        certified=False,
        normalization_common_form_separation_certified=False,
        completion_reports=(
            common,
            replace(
                split,
                constituent_identity_certified=False,
                asymptotic_convergence_certified=False,
            ),
        ),
    )
    monkeypatch.setattr(
        owner, "_assess_sine_coupling_successor", lambda **kwargs: unrelated
    )
    assert _assess().certified
    corruptions = (
        replace(source, form_error_bound=0),
        replace(source, edge_weight_deviation_upper_bound=10**9 * ETA + Q(1, 10**100)),
        replace(source, new_weight_range=(Q(1, 2), Q(1))),
        replace(
            source,
            precontact_reference=replace(
                source.precontact_reference, initial_form_means=(Q(0), Q(1, 10**100))
            ),
        ),
        replace(
            source,
            completion_reports=(
                replace(common, initial_energy_upper_bound=Q(1, 65536)),
                split,
            ),
        ),
        replace(
            source,
            completion_reports=(
                replace(common, geometry=replace(common.geometry, degree=8)),
                split,
            ),
        ),
    )
    for changed in corruptions:
        monkeypatch.setattr(
            owner,
            "_assess_sine_coupling_successor",
            lambda changed=changed, **kwargs: changed,
        )
        unavailable = _assess()
        assert not unavailable.source_certified and not unavailable.certified
        assert (
            "original_connected_source_unavailable" in unavailable.unresolved_conditions
        )
    no_existence = replace(
        source,
        completion_reports=(
            replace(common, asymptotic_convergence_certified=False),
            split,
        ),
    )
    monkeypatch.setattr(
        owner, "_assess_sine_coupling_successor", lambda **kwargs: no_existence
    )
    conditional = _assess()
    assert conditional.source_certified and conditional.conditional_ready_source
    assert (
        not conditional.finite_protocol_ready_age_exists and not conditional.certified
    )
    monkeypatch.setattr(
        owner, "_assess_sine_coupling_successor", lambda **kwargs: source
    )
    assert not _assess(form_error_bound=ERROR + Q(1, 10**100)).source_certified


def test_arithmetic_budget_is_separate_from_reference_and_reading_targets(monkeypatch):
    original = owner.metzler_exponential_bounds

    def widened(*args, **kwargs):
        result = original(*args, **kwargs)
        amount = Q(1, 10**25)
        lower = tuple(
            tuple(
                value - (amount if (i, j) == (2, 0) else 0)
                for j, value in enumerate(row)
            )
            for i, row in enumerate(result.lower)
        )
        upper = tuple(
            tuple(
                value + (amount if (i, j) == (2, 0) else 0)
                for j, value in enumerate(row)
            )
            for i, row in enumerate(result.upper)
        )
        return replace(result, lower=lower, upper=upper)

    monkeypatch.setattr(owner, "metzler_exponential_bounds", widened)
    prediction = _predict()
    assert (
        prediction.reference.gain_certified and prediction.reference.residual_certified
    )
    assert prediction.record_prediction_error_upper_bound < Q(2, 10**6)
    assert prediction.arithmetic_error_upper_bound > ARITHMETIC
    assert not prediction.arithmetic_certified and not prediction.certified
    assert "arithmetic_allowance_unavailable" in prediction.unresolved_conditions


def test_malformed_raw_inputs_stop_before_source_field_or_reference_work(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("malformed primitive reached source or numerical work")

    for name in (
        "_assess_sine_coupling_successor",
        "_coupling_successor_field",
        "metzler_exponential_bounds",
        "pi_interval",
        "I",
    ):
        monkeypatch.setattr(owner, name, forbidden)
    malformed = (
        True,
        np.bool_(False),
        "0",
        float("nan"),
        float("inf"),
        complex(0, 1),
        None,
    )
    for value in malformed:
        for key in (
            "regional_mean_reading",
            "regional_mean_error_bound",
            "readout_error_bound",
        ):
            with pytest.raises((TypeError, ValueError, OverflowError)):
                _predict(**{key: value})
        for key in (
            "form_error_bound",
            "phase_error_bound",
            "starting_gap_upper_bound",
            "readout_error_bound",
        ):
            with pytest.raises((TypeError, ValueError, OverflowError)):
                _assess(**{key: value})
        with pytest.raises((TypeError, ValueError, OverflowError)):
            owner._mixed_regional_boundary_field(
                new_edge_conductances=_weights(), stage="charge", charge_amplitude=value
            )
    for key in ("regional_mean_error_bound", "readout_error_bound"):
        with pytest.raises(ValueError):
            _predict(**{key: -Q(1, 10**100)})
    for key in (
        "form_error_bound",
        "phase_error_bound",
        "starting_gap_upper_bound",
        "readout_error_bound",
    ):
        with pytest.raises(ValueError):
            _assess(**{key: -Q(1, 10**100)})
    for stage in (True, b"probe", "PROBE", "", "uniform"):
        with pytest.raises((TypeError, ValueError)):
            owner._mixed_regional_boundary_field(
                new_edge_conductances=_weights(), stage=stage
            )
    for stage, amplitude in (
        ("probe", Q(1, 10**100)),
        ("charge", AMPLITUDE + Q(1, 10**100)),
        ("charge", -AMPLITUDE - Q(1, 10**100)),
    ):
        with pytest.raises(ValueError):
            owner._mixed_regional_boundary_field(
                new_edge_conductances=_weights(),
                stage=stage,
                charge_amplitude=amplitude,
            )


def test_shared_weight_state_and_interval_admission_preserve_region_address(mp, fields):
    for weights in (
        (_weights()[:-1]),
        _weights() + (Q(1),),
        repeat(Q(1)),
        (True,) + _weights()[1:],
        (Q(3, 4) - Q(1, 10**100),) + _weights()[1:],
        (Q(1) + Q(1, 10**100),) + _weights()[1:],
    ):
        with pytest.raises((TypeError, ValueError)):
            owner._mixed_regional_boundary_field(
                new_edge_conductances=weights, stage="probe"
            )
    state = _state((Q(0),) * 18, (Q(0),) * 18)
    for invalid in (
        state[:-1],
        state + (I(0),),
        repeat(I(0)),
        (True,) + state[1:],
        (float("nan"),) + state[1:],
        (Jet((I(0), I(1))),) + state[1:],
    ):
        with pytest.raises((TypeError, ValueError, OverflowError)):
            fields["probe"](invalid)
    jets = tuple(Jet((value, I(0))) for value in state)
    with pytest.raises((TypeError, ValueError)):
        fields["probe"]((Jet((I(0), I(0), I(1))),) + jets[1:])
    width = Q(1, 10**12)
    interval_state = (I(-width, width),) + state[1:]
    enclosed = fields["probe"](interval_state)
    for endpoint in (-width, width):
        x, _, _, intrinsic, phase = _oracle(
            mp, (endpoint,) + (Q(0),) * 17, (Q(0),) * 18
        )
        expected = (
            tuple(value - (x[i] if i < 9 else 0) for i, value in enumerate(intrinsic))
            + phase
        )
        for interval, value in zip(enclosed, expected):
            assert _mp(mp, interval.lo) <= value <= _mp(mp, interval.hi)
    assert sum(enclosed[:18], I(0)).lo <= -width <= width <= sum(enclosed[:18], I(0)).hi


def test_one_shot_fixed_width_and_decoder_inputs_do_not_supply_hidden_response(
    fields, report
):
    def fixed(value):
        return Q(np.int64(Q(value).numerator), np.int64(Q(value).denominator))

    field = owner._mixed_regional_boundary_field(
        new_edge_conductances=(fixed(value) for value in _weights()),
        stage="charge",
        charge_amplitude=fixed(AMPLITUDE),
    )
    state = _state((Q(0),) * 18, (Q(0),) * 18)
    assert field(value for value in state) == fields["charge"](state)
    ordinary = _predict(regional_mean_reading=-MEAN)
    normalized = _predict(
        regional_mean_reading=fixed(-MEAN),
        regional_mean_error_bound=fixed(NOISE),
        readout_error_bound=fixed(NOISE),
    )
    assert ordinary == normalized
    assert (
        _assess(
            form_error_bound=fixed(ERROR),
            phase_error_bound=fixed(ERROR),
            starting_gap_upper_bound=fixed(DELTA / 8),
            readout_error_bound=fixed(NOISE),
        )
        == report
    )
    with pytest.raises(FrozenInstanceError):
        ordinary.forecast = Q(0)
    with pytest.raises(FrozenInstanceError):
        report.norm_tube_upper_bound = Q(0)
    for key in (
        "global_mean_after_probe",
        "global_history",
        "regional_current",
        "reserved_response",
        "charge_history",
        "source_state",
        "elapsed_time",
        "equilibrium",
        "reference",
    ):
        with pytest.raises(TypeError):
            _predict(**{key: 0})
    for key in ("ready_age", "source", "equilibrium", "horizon"):
        with pytest.raises(TypeError):
            _assess(**{key: 0})
