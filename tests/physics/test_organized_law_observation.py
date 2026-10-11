"""Static own-law controls; no record, forecast or source producer is evaluated."""

from fractions import Fraction as Q

import mpmath
import pytest

from tnfr.mathematics._interval_taylor import Jet
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.physics._organized_computable_prediction import _ForecastStage
from tnfr.physics._organized_law_observation import (
    _admit_carrier_stage,
    _bound_organized_observation_law,
    _organized_law_carrier_field,
)
from tnfr.physics._sine_class_constitutive_comparison import _edge_law


@pytest.mark.parametrize(
    "bad", [True, False, float("nan"), float("inf"), -Q(1, 10), Q(11, 10)]
)
def test_deformation_is_admitted_before_law_construction(bad):
    with pytest.raises((TypeError, ValueError)):
        _bound_organized_observation_law(deformation=bad)
    with pytest.raises((TypeError, ValueError)):
        _organized_law_carrier_field(deformation=bad)


@pytest.mark.parametrize("coefficient", [Q(0), Q(1, 3), Q(1)])
def test_own_law_bounds_rebuild_integral_comparisons(coefficient):
    result = _bound_organized_observation_law(deformation=coefficient)
    assert result.current_upper == 1 + coefficient / 1458
    assert result.current_lipschitz == 1 + 2 * coefficient / 243
    assert result.current_second_derivative_upper == 1 + coefficient / 9
    hidden = result.hidden
    coupling = 2 * result.gamma_upper * result.duration * result.current_lipschitz
    assert hidden.form_upper == hidden.initial_form + coupling * hidden.phase_upper
    assert hidden.phase_upper == hidden.initial_phase + coupling * hidden.form_upper
    residual = (
        result.gamma_upper
        * result.current_second_derivative_upper
        * hidden.phase_upper**2
        / 9
    )
    k = result.gamma_upper * result.duration
    assert result.source_form_error_upper == (
        result.duration * residual
        + result.current_lipschitz * k**2 * result.source_form_error_upper
    )
    assert result.source_phase_error_upper == k * result.source_form_error_upper
    assert hidden.phase_upper < Q(605, 10**10)
    assert result.source_form_error_upper < Q(242, 10**20)
    assert result.carrier_log_norm_upper == (1 + result.current_lipschitz) / (2 * 3069)
    assert result.carrier_growth_margin > 0
    assert all(value > 0 for _, value in result.identity_margins)
    assert dict(result.identity_margins)["outer_acute_gap"] > Q(69, 1000)
    assert result.admitted


def test_sine_endpoint_recovers_existing_source_bound_without_transferring_alternative():
    from tnfr.physics._organized_observed_prediction import _observation_budget

    sine = _bound_organized_observation_law(deformation=0)
    alternative = _bound_organized_observation_law(deformation=1)
    retained = _observation_budget()
    assert sine.hidden.phase_upper == retained.hidden_phase_upper
    assert sine.source_form_error_upper == retained.source_error
    assert alternative.source_form_error_upper > sine.source_form_error_upper
    assert alternative.fine_phase_drift_upper > sine.fine_phase_drift_upper


@pytest.fixture(scope="module")
def context():
    result = mpmath.mp.clone()
    result.dps = 90
    return result


def _mp(context, value):
    return context.mpf(value.numerator) / value.denominator


def _encloses(context, interval, value):
    assert _mp(context, interval.lo) <= value <= _mp(context, interval.hi)


@pytest.mark.parametrize("coefficient", [Q(0), Q(1, 2), Q(1)])
def test_carrier_equals_independent_full_fine_projection(context, coefficient):
    form = tuple(Q(i - 2, 1000) for i in range(5))
    phase = tuple(Q(i, 10) for i in (0, 3, -2, 7, -4))
    fine_form = tuple(_mp(context, form[c]) for c in range(5) for _ in range(9))
    fine_phase = tuple(
        2 * context.pi * (j - 4) / 9 + _mp(context, phase[c])
        for c in range(5)
        for j in range(9)
    )
    inner = tuple((9 * c + j, 9 * c + (j + 1) % 9) for c in range(5) for j in range(9))
    outer = tuple(
        (9 * c + j, 9 * ((c + 1) % 5) + j) for c in range(5) for j in range(9)
    )
    gradient, currents = [context.mpf(0)] * 45, [context.mpf(0)] * 45
    for tail, head in inner + outer:
        difference = fine_form[tail] - fine_form[head]
        angle = fine_phase[head] - fine_phase[tail]
        current = context.sin(angle) + _mp(context, coefficient) / 2187 * (
            context.sin(9 * angle) - context.sin(18 * angle) / 2
        )
        gradient[tail] += difference
        gradient[head] -= difference
        currents[tail] += current
        currents[head] -= current
    gamma = 1 / (1023 * context.pi)
    form_rates = tuple(
        (-q + gamma * current) / 4 for q, current in zip(gradient, currents)
    )
    phase_rates = tuple(gamma * q / 4 for q in gradient)
    flow = _organized_law_carrier_field(deformation=coefficient)
    actual = flow(I(0), tuple(I(value) for value in form + phase))
    for child in range(5):
        mean_form = sum(form_rates[9 * child : 9 * child + 9]) / 9
        mean_phase = sum(phase_rates[9 * child : 9 * child + 9]) / 9
        _encloses(context, actual[child], mean_form)
        _encloses(context, actual[5 + child], mean_phase)
        # Every slot has the same rate: this is an invariant carrier, not
        # merely an accidentally matching instantaneous arithmetic mean.
        assert max(
            abs(rate - mean_form) for rate in form_rates[9 * child : 9 * child + 9]
        ) < context.mpf("1e-80")
        assert max(
            abs(rate - mean_phase) for rate in phase_rates[9 * child : 9 * child + 9]
        ) < context.mpf("1e-80")


def test_constitutive_target_matching_does_not_transfer_outer_tangent(context):
    for angle, equal in ((2 * context.pi / 9, True), (2 * context.pi / 5, False)):
        sine = _edge_law(context.cos(angle), context.sin(angle), Q(0))[1]
        altered = _edge_law(context.cos(angle), context.sin(angle), Q(1))[1]
        if equal:
            assert abs(altered[1] - sine[1]) < context.mpf("1e-80")
        else:
            assert altered[1] - sine[1] > context.mpf("0.004")


def test_shared_stage_shift_and_clock_endpoints_preserve_actual_rows():
    stage = _ForecastStage(Q(200), Q(1, 100000), "plateau", 2, Q(1, 10))
    commanded = _organized_law_carrier_field(deformation=1, stage=stage)
    unforced = _organized_law_carrier_field(deformation=1)
    state = tuple(I(Q(i, 1000)) for i in range(10))
    time = I(stage.start + stage.duration)
    command = stage.command(time)
    shifted = state[:5] + tuple(
        value + offset for value, offset in zip(state[5:], command)
    )
    assert commanded(time, state) == unforced(time, shifted)
    with pytest.raises(ValueError, match="outside"):
        commanded(I(201), state)


@pytest.mark.parametrize(
    "attribute,bad",
    [
        ("start", True),
        ("start", float("inf")),
        ("duration", False),
        ("duration", float("nan")),
        ("duration", Q(0)),
        ("duration", Q(-1)),
        ("child", True),
        ("child", False),
        ("child", 5),
        ("child", -1),
        ("child", 2.0),
        ("base", True),
        ("base", float("inf")),
        ("increment", False),
        ("increment", float("nan")),
    ],
)
def test_stage_primitives_reject_before_geometry_materialization(
    attribute, bad, monkeypatch
):
    from dataclasses import replace
    from tnfr.physics import _organized_law_observation as owner

    stage = replace(_ForecastStage(Q(200), Q(1), "ramp", 2), **{attribute: bad})

    def forbidden():
        pytest.fail("invalid stage reached geometry materialization")

    monkeypatch.setattr(owner, "_derive_organized_outer_winding", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _organized_law_carrier_field(deformation=1, stage=stage)


def test_stage_rebuild_preserves_label_and_does_not_admit_unconsumed_coordinates():
    raw = _ForecastStage(200.0, 1.0, "arbitrary-label", 2, 0.1, -0.2)
    normalized = _admit_carrier_stage(raw)
    assert normalized == _ForecastStage(
        Q(200), Q(1), raw.name, 2, Q.from_float(0.1), Q.from_float(-0.2)
    )
    assert type(normalized.start) is type(normalized.duration) is Q
    unused = _ForecastStage(Q(200), Q(1), "ramp", None, True, float("nan"))
    normalized_unused = _admit_carrier_stage(unused)
    assert normalized_unused.base is unused.base
    assert normalized_unused.increment is unused.increment
    state = tuple(I(i) for i in range(10))
    assert _organized_law_carrier_field(deformation=1, stage=unused)(I(200), state) == (
        _organized_law_carrier_field(deformation=1)(I(200), state)
    )
    changed_label = _ForecastStage(
        raw.start, raw.duration, "formation", raw.child, raw.base, raw.increment
    )
    assert _organized_law_carrier_field(deformation=1, stage=raw)(I(200), state) == (
        _organized_law_carrier_field(deformation=1, stage=changed_label)(I(200), state)
    )


def test_commanded_jets_require_clock_order_but_allow_held_time_state_derivatives():
    stage = _ForecastStage(Q(200), Q(1), "ramp", 2, Q(0), Q(1, 10))
    flow = _organized_law_carrier_field(deformation=1, stage=stage)
    state = tuple(Jet((I(0), I(0))) for _ in range(10))
    held = Jet((I(Q(401, 2)), I(0)))
    progressing = Jet((I(Q(401, 2)), I(1)))
    assert len(flow(held, state)) == 10
    assert any(
        value.coeffs[1].lo > Q(1, 10**8) or value.coeffs[1].hi < -Q(1, 10**8)
        for value in flow(progressing, state)[:5]
    )
    for wrong_clock in (I(Q(401, 2)), Jet.constant(Q(401, 2), 2)):
        with pytest.raises(ValueError, match="clock jet"):
            flow(wrong_clock, state)
    with pytest.raises(ValueError, match="interval command clock"):
        flow(progressing, (I(0),) * 10)
    assert len(flow(I(Q(401, 2)), (I(0),) * 10)) == 10


def test_jets_retain_full_nodal_rows_and_reject_invented_coordinates():
    flow = _organized_law_carrier_field(deformation=1)
    state = tuple(Jet((I(int(i == 0)), I(0))) for i in range(10))
    actual = flow(Jet((I(0), I(1))), state)
    gradient = (2, -1, 0, 0, -1)
    gamma = Q(1, 1023) / pi_interval()
    for i, value in enumerate(gradient):
        assert actual[i].coeffs[0].lo <= -Q(value, 4) <= actual[i].coeffs[0].hi
        assert actual[i + 5].coeffs[0] == value * gamma / 4
    with pytest.raises(ValueError, match="five"):
        flow(I(0), (I(0),) * 9)
    with pytest.raises(ValueError, match="common order"):
        flow(I(0), (I(0),) + state[1:])
    with pytest.raises(TypeError):
        flow(I(0), (True,) + (I(0),) * 9)
