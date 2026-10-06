"""Independent full-node and admission controls for moving work ports."""

import json
from dataclasses import replace
from fractions import Fraction as Q

import mpmath as mp
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._validated_taylor import flow_jets
from tnfr.physics import relational_sine_scale as scale
from tnfr.sdk import export_to_json, relational_report_to_dict

MODEL = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
ARGS = dict(
    reference_model=MODEL,
    form_half_difference=Q(1, 32),
    phase_half_difference=Q(1, 32),
    capacity=1,
    scaled_duration=Q(1, 16),
    order=10,
)
PAIRS = tuple((2 * j, 2 * j + 1) for j in range(5))
EDGES = tuple((i, k) for j in range(5) for i in PAIRS[j] for k in PAIRS[(j + 1) % 5])


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(bound, value):
    if bound.lo == bound.hi:
        assert abs(value - _mp(bound.lo)) < mp.mpf("1e-75")
    else:
        assert _mp(bound.lo) <= value <= _mp(bound.hi)


@pytest.fixture(scope="module")
def witness():
    """One frozen endpoint plus its shared field, reused by all controls."""
    captured = {}
    original = scale.validated_taylor_step

    def capture(initial, duration, field, domain, **kwargs):
        captured.update(field=field, initial=initial, domain=domain)
        return original(initial, duration, field, domain, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(scale, "validated_taylor_step", capture)
        report = scale.assess_sine_replica_pulse_work_response(**ARGS)
    return report, captured


def _trig_coefficients(theta, order):
    """Independent normalized power-series recurrence, not production jets."""
    sine, cosine = [mp.sin(theta[0])], [mp.cos(theta[0])]
    for n in range(1, order + 1):
        sine.append(sum(k * theta[k] * cosine[n - k] for k in range(1, n + 1)) / n)
        cosine.append(-sum(k * theta[k] * sine[n - k] for k in range(1, n + 1)) / n)
    return sine, cosine


def _fine_taylor(order=6):
    """All twenty nonlinear fine rows and both twenty-row variations."""
    alpha, u, delta = 2 * mp.pi / 5, mp.mpf(1) / 32, mp.mpf(1) / 32
    x = [[u if i % 2 == 0 else -u] for i in range(10)]
    theta = [[(i // 2) * alpha + (delta if i % 2 == 0 else -delta)] for i in range(10)]
    ports = (
        tuple(mp.cos(alpha * (i // 2)) for i in range(10)),
        tuple((1 if i % 2 == 0 else -1) * mp.sin(alpha * (i // 2)) for i in range(10)),
    )
    dx = [[[f] for f in port] for port in ports]
    dtheta = [[[mp.mpf(0)] for _ in range(10)] for _ in ports]
    for n in range(order):
        form_rate = [mp.mpf(0)] * 10
        phase_rate = [mp.mpf(0)] * 10
        var_form = [[mp.mpf(0)] * 10 for _ in ports]
        var_phase = [[mp.mpf(0)] * 10 for _ in ports]
        for i, j in EDGES:
            gap = [theta[j][k] - theta[i][k] for k in range(n + 1)]
            sine, cosine = _trig_coefficients(gap, n)
            form_rate[i] += sine[n] / 4
            form_rate[j] -= sine[n] / 4
            phase_rate[i] += (x[i][n] - x[j][n]) / 4
            phase_rate[j] -= (x[i][n] - x[j][n]) / 4
            for col in range(2):
                change = (
                    sum(
                        cosine[k] * (dtheta[col][j][n - k] - dtheta[col][i][n - k])
                        for k in range(n + 1)
                    )
                    / 4
                )
                var_form[col][i] += change
                var_form[col][j] -= change
                phase_change = (dx[col][i][n] - dx[col][j][n]) / 4
                var_phase[col][i] += phase_change
                var_phase[col][j] -= phase_change
        for i in range(10):
            x[i].append(form_rate[i] / (n + 1))
            theta[i].append(phase_rate[i] / (n + 1))
            for col in range(2):
                dx[col][i].append(var_form[col][i] / (n + 1))
                dtheta[col][i].append(var_phase[col][i] / (n + 1))
    outputs = tuple(
        tuple(
            tuple(
                sum(
                    (ports[row][i] - ports[row][j]) * (dx[col][i][n] - dx[col][j][n])
                    for i, j in EDGES
                )
                for n in range(order + 1)
            )
            for col in range(2)
        )
        for row in range(2)
    )
    return ports, outputs


def test_full_twenty_coordinate_jets_and_actual_work_port_normalization(witness):
    report, captured = witness
    coefficients = flow_jets(captured["initial"], 6, captured["field"])
    with mp.workdps(90):
        ports, expected = _fine_taylor()
        l = 1 - mp.cos(2 * mp.pi / 5)
        for port in ports:
            assert abs(sum(v * v for v in port) - 5) < mp.mpf("1e-80")
        _contains(report.work_output_coefficients[0], 20 * l)
        _contains(report.work_output_coefficients[1], mp.mpf(20))
        for row in range(2):
            for col, start in enumerate((2, 6)):
                for n in range(7):
                    _contains(
                        report.work_output_coefficients[row]
                        * coefficients[start + 2 * row][n],
                        expected[row][col][n],
                    )
        # No antisymmetry appears until the time-ordering commutator at order five.
        for n in range(5):
            assert abs(expected[0][1][n] - expected[1][0][n]) < mp.mpf("1e-75")
        c = mp.cos(2 * mp.pi / 5)
        s = mp.sin(2 * mp.pi / 5)
        commutator_work = (
            20
            * l
            * mp.mpf(1)
            / 32
            * c
            * s**2
            * (1 - l)
            * (1 + l * mp.cos(mp.mpf(1) / 32) ** 2)
        )
        assert abs(
            expected[0][1][5] - expected[1][0][5] - commutator_work / 60
        ) < mp.mpf("1e-75")


def test_frozen_budget_excludes_zero_with_independent_volterra_sign_bound(witness):
    report, _ = witness
    h = ARGS["scaled_duration"]
    # Whole-pulse interval inequalities bound the ordered double integral;
    # the remaining even Volterra terms use ||C D||_infinity <= 3/8.
    leading = Q(33, 800000) * h**5
    tail = 2 * Q(3, 8) ** 3 * h**6 / (720 * (1 - Q(3, 8) * h**2 / 56))
    independent_lower = 20 * (leading - tail)
    assert independent_lower > 0
    assert report.status == "certified_finite_response"
    assert report.directional_response_certified
    assert report.response_sign == 1
    assert report.antisymmetric_response_bounds.lo > independent_lower
    assert report.step.picard_interior_margin > 0
    assert report.step.duration == h
    assert report.clock == "tau=t/pi"
    assert report.reference.full_real_dimension == 20


def test_freezing_the_same_generator_restores_matched_port_reciprocity():
    with mp.workdps(90):
        alpha, delta = 2 * mp.pi / 5, mp.mpf(1) / 32
        c, s, l = mp.cos(alpha), mp.sin(alpha), 1 - mp.cos(alpha)
        C, S = mp.cos(delta), mp.sin(delta)
        A = mp.matrix(
            [
                [0, -c * C**2 * l, 0, -(s**2) * C * S],
                [l, 0, 0, 0],
                [0, -(s**2) * C * S, 0, -c * (1 - (1 + c) * S**2)],
                [0, 0, 1, 0],
            ]
        )
        frozen = mp.expm(A / 16)
        assert abs(20 * l * frozen[0, 2] - 20 * frozen[2, 0]) < mp.mpf("1e-75")


@pytest.mark.parametrize(
    "u,delta", [(0, 0), (Q(-1, 32), Q(1, 32)), (Q(-1, 32), Q(-1, 32))]
)
def test_static_motion_reversal_and_member_swap_controls(witness, u, delta):
    original, _ = witness
    report = scale.assess_sine_replica_pulse_work_response(
        **{**ARGS, "form_half_difference": u, "phase_half_difference": delta}
    )
    assert report.status == "certified_finite_response"
    if u == delta == 0:
        assert report.response_sign == 0
        assert not report.directional_response_certified
        assert (
            report.antisymmetric_response_bounds.lo
            <= 0
            <= report.antisymmetric_response_bounds.hi
        )
    else:
        assert report.response_sign == -1
        assert report.antisymmetric_response_bounds.hi < 0
        if delta < 0:
            # Co-swapping the internal port restores the same experiment.
            for row in range(2):
                for col in range(2):
                    transformed = report.response_bounds[row][col] * (
                        -1 if row != col else 1
                    )
                    expected = original.response_bounds[row][col]
                    assert max(transformed.lo, expected.lo) <= min(
                        transformed.hi, expected.hi
                    )


def test_declared_weight_capacity_and_beta_remain_in_the_actual_scaled_field(
    monkeypatch,
):
    captured = {}

    def capture(initial, duration, field, domain, **kwargs):
        captured.update(initial=initial, field=field)
        return None, initial, "instrumented_no_evaluation"

    monkeypatch.setattr(scale, "validated_taylor_step", capture)
    model = replace(MODEL)
    object.__setattr__(model, "phase_weight", Q(3, 2))
    object.__setattr__(model, "storage_scale", Q(5, 4))
    report = scale.assess_sine_replica_pulse_work_response(
        **{**ARGS, "reference_model": model, "capacity": Q(2, 3)}
    )
    rates = captured["field"](captured["initial"])
    with mp.workdps(90):
        delta = mp.mpf(1) / 32
        c = mp.cos(2 * mp.pi / 5)
        _contains(rates[0], -c * mp.cos(delta) * mp.sin(delta))
        _contains(rates[1], mp.mpf(1) / 40)
        _contains(rates[3], mp.mpf(4) / 5 * (1 - c))
        _contains(rates[9], mp.mpf(4) / 5)
    assert report.status == "unavailable"
    assert report.response_bounds is None
    assert report.response_sign is None
    assert not report.directional_response_certified
    assert report.unavailable_reasons == ("instrumented_no_evaluation",)


def test_actual_chart_exit_returns_unavailable_without_retrying_the_budget():
    # Conserved u^2+c*sin(delta)^2=1 keeps u>=sqrt(1-c), so this
    # rotating preparation reaches delta=pi/2 before the declared horizon.
    report = scale.assess_sine_replica_pulse_work_response(
        **{
            **ARGS,
            "form_half_difference": 1,
            "phase_half_difference": 0,
            "scaled_duration": 4,
        }
    )
    assert report.scaled_duration == 4
    assert report.taylor_order == 10
    assert report.status == "unavailable"
    assert report.step is None
    assert report.failed_tube is not None
    assert report.response_bounds is None
    assert report.antisymmetric_response_bounds is None
    assert report.response_sign is None
    assert not report.directional_response_certified
    assert report.unavailable_reasons


@pytest.mark.parametrize(
    "changes",
    [
        {"scaled_duration": 0},
        {"scaled_duration": -1},
        {"scaled_duration": True},
        {"scaled_duration": float("nan")},
        {"order": True},
        {"order": 0},
        {"order": 1000},
        {"capacity": 0},
        {"capacity": False},
        {"form_half_difference": True},
        {"phase_half_difference": float("inf")},
        {"phase_half_difference": 2},
        {"reference_model": None},
        {"reference_model": RelationalExchangeModel(1, phase_domain="regular")},
    ],
)
def test_invalid_primitives_reject_before_numerical_evaluation(monkeypatch, changes):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid primitive reached numerical propagation")

    monkeypatch.setattr(scale, "validated_taylor_step", forbidden)
    with pytest.raises((TypeError, ValueError)):
        scale.assess_sine_replica_pulse_work_response(**{**ARGS, **changes})


def test_boolean_model_forgery_cannot_use_python_numeric_equality():
    model = replace(MODEL)
    object.__setattr__(model, "phase_weight", True)
    with pytest.raises((TypeError, ValueError)):
        scale.assess_sine_replica_pulse_work_response(
            **{**ARGS, "reference_model": model}
        )


def test_shared_sdk_export_keeps_exact_budget_and_derivative_scope(witness, tmp_path):
    report, _ = witness
    data = relational_report_to_dict(report)
    assert data["report"] == report.to_dict()["report"]
    assert (
        report.to_dict()["schema"]
        == "tnfr.relational-sine-replica-pulse-work-response.v1"
    )
    assert data["schema"] == "tnfr.relational-report.v1"
    assert data["report_type"] == "SineReplicaPulseWorkResponse"
    assert data["report"]["scaled_duration"] == {"numerator": 1, "denominator": 16}
    assert data["report"]["directional_response_certified"] is True
    assert (
        "response_is_derivative_at_zero_kick_with_unperturbed_motion_subtracted"
        in data["report"]["scope"]
    )
    target = tmp_path / "moving-work-response.json"
    export_to_json(data, target)
    assert json.loads(target.read_text(encoding="utf-8")) == data
