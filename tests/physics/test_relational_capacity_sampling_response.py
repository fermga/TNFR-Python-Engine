"""Synthetic algebra and protocol admission, without reserved K3 evaluation.

All endpoint calculations below replace the candidate flow by explicit test
polynomials. Actual reserved responses belong to the frozen producer record.
"""

from dataclasses import replace
from fractions import Fraction as Q
from math import factorial

import pytest

from tnfr.mathematics._interval_taylor import MAX_ORDER
from tnfr.mathematics._rational_interval import I
from tnfr.physics.relational_observations import bound_relational_rate_from_samples
from tnfr.research import relational_capacity_discriminator as owner


@pytest.fixture(scope="module")
def protocol():
    # Prior domain/remainder feasibility only; this factory samples no response.
    return owner.prepare_relational_capacity_response()


@pytest.mark.parametrize("order", (0, -1, True, None, 1.5, MAX_ORDER + 1))
def test_flow_jet_order_rejects_before_consuming_a_field(monkeypatch, order):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid order must reject before field evaluation")

    monkeypatch.setattr(owner, "_field_jets", forbidden)
    with pytest.raises(ValueError, match="order"):
        owner._flow_coefficients(
            (I(0),) * 6, (Q(1),) * 3, I(1), mediated=False, order=order
        )


def test_higher_flow_orders_reuse_normalized_series_recurrence(monkeypatch):
    def synthetic_decay(state, *args, **kwargs):
        return tuple(-value for value in state)

    monkeypatch.setattr(owner, "_field_jets", synthetic_decay)
    result = owner._flow_coefficients(
        (I(1),) * 6, (Q(1),) * 3, I(1), mediated=False, order=7
    )
    assert all(len(row) == 8 for row in result)
    for row in result:
        for degree, value in enumerate(row):
            assert value.contains(Q((-1) ** degree, factorial(degree)))


def test_prospective_factory_has_complete_fixed_information_without_response(protocol):
    assert protocol.source_beta == 1 and protocol.taylor_order == 6
    assert len(protocol.whole_window_remainder_bounds) == 4
    assert all(
        0 <= bound < protocol.sampling.sample_error_bound
        for _, _, bound in protocol.whole_window_remainder_bounds
    )
    predictions = dict(protocol.predictions)
    assert predictions["capacity_separable"] == (0, 0)
    assert -Q(81, 1000) < predictions["capacity_mediated"][0]
    assert predictions["capacity_mediated"][1] < -Q(8, 100)
    assert protocol.arithmetic_allowance == Q(1, 1584)
    assert protocol.envelope_radius == Q(1, 48)


@pytest.mark.parametrize(
    "change",
    (
        {"source_beta": Q(2)},
        {"source_beta": True},
        {"taylor_order": 7},
        {"arithmetic_allowance": Q(1)},
    ),
)
def test_changed_protocol_rejects_before_any_response(monkeypatch, protocol, change):
    monkeypatch.setattr(owner, "prepare_relational_capacity_response", lambda: protocol)

    def forbidden(*args, **kwargs):
        raise AssertionError("changed protocol cannot evaluate a response")

    monkeypatch.setattr(owner, "_response_case", forbidden)
    with pytest.raises(ValueError, match="prospective declaration"):
        owner.evaluate_relational_capacity_response(replace(protocol, **change))


def test_response_requires_a_typed_protocol():
    with pytest.raises(TypeError, match="Protocol"):
        owner.evaluate_relational_capacity_response({})


def _synthetic_coefficients(box, capacity, beta, *, mediated, order=3):
    rows = [[I(0)] * (order + 1) for _ in range(6)]
    if order == 6:
        rows[3][:4] = [I(1), I(2), I(3), I(4)]
    elif order == 7:
        rows[3][7] = I(-5, 5)
    else:
        raise AssertionError("this fixture only supplies the declared two orders")
    return tuple(tuple(row) for row in rows)


def test_synthetic_polynomial_keeps_whole_box_remainder_without_extra_factorial(
    monkeypatch, protocol
):
    monkeypatch.setattr(owner, "_flow_coefficients", _synthetic_coefficients)
    report = owner._response_case(protocol, protocol.sampling.cases[0])
    assert report.status == "completed"
    assert len(report.initial_coefficients) == 6
    assert len(report.initial_coefficients[0]) == 7
    assert report.remainder_coefficients[3] == (-5, 5)
    for sample in report.samples:
        time = sample.time
        polynomial = 1 + 2 * time + 3 * time**2 + 4 * time**3
        radius = 5 * time**7
        assert sample.polynomial_bounds[3] == (polynomial, polynomial)
        assert sample.remainder_bounds[3] == (-radius, radius)
        assert sample.phase_contrast_bounds == (
            polynomial - radius,
            polynomial + radius,
        )
        assert sample.midpoint == polynomial and sample.radius == radius
    observation = report.rate_observation
    assert observation.sample_error_bound == protocol.sampling.sample_error_bound
    assert (
        observation.third_derivative_bound == protocol.sampling.third_derivative_bound
    )
    assert observation.rate_error_bound == protocol.sampling.rate_error_bound
    # The fixed error budget is not silently replaced by smaller enclosure radii.
    assert observation.sample_error_bound > max(
        sample.radius for sample in report.samples
    )


def test_synthetic_excess_width_retains_first_failure_and_partial_case(
    monkeypatch, protocol
):
    def wide(*args, **kwargs):
        rows = [list(row) for row in _synthetic_coefficients(*args, **kwargs)]
        if kwargs["order"] == 6:
            rows[3][0] = I(0, 2)
        return tuple(tuple(row) for row in rows)

    monkeypatch.setattr(owner, "_flow_coefficients", wide)
    report = owner._response_case(protocol, protocol.sampling.cases[0])
    assert report.status == "stopped"
    assert len(report.samples) == 1
    assert report.samples[0].radius == 1
    assert not report.samples[0].admitted
    assert report.initial_coefficients and report.remainder_coefficients
    assert report.rate_observation is None
    assert report.unavailable_reasons == (
        "sample_enclosure_exceeds_frozen_error_budget",
    )


def test_synthetic_failure_preserves_initial_coefficients_without_retry(
    monkeypatch, protocol
):
    calls = []

    def fail_remainder(*args, **kwargs):
        calls.append(kwargs["order"])
        if kwargs["order"] == 7:
            raise ArithmeticError("synthetic remainder failure")
        return _synthetic_coefficients(*args, **kwargs)

    monkeypatch.setattr(owner, "_flow_coefficients", fail_remainder)
    report = owner._response_case(protocol, protocol.sampling.cases[0])
    assert calls == [6, 7]
    assert report.status == "stopped"
    assert report.initial_coefficients
    assert report.remainder_coefficients == report.samples == ()
    assert report.unavailable_reasons == (
        "response_stopped: ArithmeticError: synthetic remainder failure",
    )


def test_full_evaluator_keeps_a_synthetic_stopped_case_and_negative_verdict(
    monkeypatch, protocol
):
    monkeypatch.setattr(owner, "prepare_relational_capacity_response", lambda: protocol)
    calls = []

    def stopped(declaration, case):
        calls.append(case)
        return owner.RelationalCapacityResponseCase(
            law=case.law,
            arm=case.arm,
            capacity=case.capacity,
            status="stopped",
            initial_coefficients=(),
            remainder_coefficients=(),
            samples=(),
            rate_observation=None,
            unavailable_reasons=("synthetic failure",),
        )

    monkeypatch.setattr(owner, "_response_case", stopped)
    report = owner.evaluate_relational_capacity_response(protocol)
    assert len(calls) == len(report.cases) == 1
    assert not report.completed and not report.passed and not report.contrasts_disjoint
    assert report.cases[0].unavailable_reasons == ("synthetic failure",)
    assert "source_incomplete" in report.unavailable_reasons
    assert all(not analysis.passed for analysis in report.analyses)
    payload = report.to_dict()
    assert payload["report"]["passed"] is False
    assert payload["report"]["cases"][0]["unavailable_reasons"] == ["synthetic failure"]


def test_fixed_positive_orientation_rejects_synthetic_negative_baseline(protocol):
    observation = bound_relational_rate_from_samples(
        (0, -protocol.sampling.sample_step, -protocol.sampling.horizon),
        sample_step=protocol.sampling.sample_step,
        sample_error_bound=protocol.sampling.sample_error_bound,
        third_derivative_bound=protocol.sampling.third_derivative_bound,
    )
    cases = tuple(
        owner.RelationalCapacityResponseCase(
            law="capacity_separable",
            arm=arm,
            capacity=(Q(1),) * 3,
            status="completed",
            initial_coefficients=(),
            remainder_coefficients=(),
            samples=(),
            rate_observation=observation,
            unavailable_reasons=(),
        )
        for arm in ("before", "after")
    )
    analysis = owner._response_analysis(protocol, "capacity_separable", cases)
    assert analysis.contrast.normalized_change_bounds is not None
    assert analysis.baseline_lower_bound < 0
    assert not analysis.passed
    assert "resolved_baseline" in analysis.unavailable_reasons
