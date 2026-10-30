"""Prospective budget disposition and wiring; no reserved IVP is evaluated."""

from fractions import Fraction as Q
from types import SimpleNamespace

import pytest

from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.research import relational_seeded_response as owner


def test_budget_preparation_retains_law_state_and_numerical_budget(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("preparation must not evaluate a response")

    monkeypatch.setattr(owner, "certify_relational_reflected_transit", forbidden)
    short = owner.prepare_relational_seeded_response()
    budget = owner.prepare_relational_seeded_response(study="target-budget")
    for key in (
        "nodes",
        "edges",
        "original_phase_pi",
        "form",
        "capacity",
        "coordinate_order",
        "initial_form_coordinates",
        "initial_phase_coordinates_pi",
        "initial_interpretation",
        "phase_frame",
        "model",
        "clock",
        "time_step",
        "order",
        "interval_bits",
        "max_endpoint_width",
        "picard_policy",
        "comparison_policy",
        "failure_policy",
    ):
        assert budget[key] == short[key]
    assert Q(**budget["horizon"]) == 1
    assert Q(**short["horizon"]) == Q(1, 8)
    assert "not_formation" in budget["passed_means"]


@pytest.mark.parametrize(
    "budget,admitted,width,sign,verdict,passed,excluded",
    (
        (I(-2, -1), True, Q(0), "negative", "target_excluded", True, True),
        (
            I(1, 2),
            True,
            Q(0),
            "positive",
            "target_not_excluded_by_storage",
            True,
            False,
        ),
        (I(-1, 1), True, Q(0), "undecided", "undecided", False, False),
        (I(-1, 0), True, Q(0), "undecided", "undecided", False, False),
        (I(0, 1), True, Q(0), "undecided", "undecided", False, False),
        (I(0), True, Q(0), "undecided", "undecided", False, False),
        (I(-2, -1), False, Q(0), "negative", "unavailable_horizon", False, True),
        (I(1, 2), False, Q(0), "positive", "unavailable_horizon", False, False),
        (I(-2, -1), True, Q(1), "negative", "unavailable_width", False, True),
    ),
)
def test_budget_sign_is_not_a_formation_success(
    monkeypatch, budget, admitted, width, sign, verdict, passed, excluded
):
    calls = []
    report = SimpleNamespace(
        endpoint=(I(0, width),) * 8,
        endpoint_source_argument=I(-1),
        initial_source_argument=I(-2),
        target_budget=budget,
        admitted=admitted,
        observations=(),
        validated_horizon=Q(1) if admitted else Q(1, 2),
        to_dict=lambda: {"fixture": "synthetic certificate, no trajectory"},
    )

    def certificate(initial, **kwargs):
        calls.append((initial, kwargs))
        return report

    monkeypatch.setattr(owner, "certify_relational_reflected_transit", certificate)
    declaration = owner.prepare_relational_seeded_response(study="target-budget")
    response = owner.evaluate_relational_seeded_response(
        declaration, study="target-budget"
    )
    assert len(calls) == 1
    initial, settings = calls[0]
    pi = pi_interval()
    assert initial == (I(0),) * 4 + (4 * pi / 5, 2 * pi / 5, I(0), I(0))
    assert settings["horizon"] == 1 and settings["time_step"] == Q(1, 64)
    assert settings["order"] == 6
    assert settings["model"].phase_domain == "regular"
    assert settings["model"].effective_weights == (Q(1, 2), Q(1, 2))
    assert settings["model"].storage_scale == 1
    assert response["budget_sign_at_validated_horizon"] == sign
    assert response["target_budget_verdict"] == verdict
    assert response["passed"] is passed
    assert response["future_target_excluded_from_validated_horizon"] is excluded
    assert response["gates"]["horizon_admitted"] is admitted
    # Winding/argument signs are recorded by the certificate, not gates that
    # would bias this open budget test toward the earlier short response.
    assert set(response["gates"]) == {
        "horizon_admitted",
        "endpoint_width_budget",
        "strict_budget_sign",
    }


@pytest.mark.parametrize("change", ("study", "horizon", "prediction"))
def test_changed_budget_declaration_rejects_before_any_flow(monkeypatch, change):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid declaration must not evolve")

    monkeypatch.setattr(owner, "certify_relational_reflected_transit", forbidden)
    declaration = owner.prepare_relational_seeded_response(study="target-budget")
    study = "target-budget"
    if change == "study":
        study = "short"
    elif change == "horizon":
        declaration["horizon"]["numerator"] += 1
    else:
        declaration["prediction"].append("posthoc condition")
    with pytest.raises(ValueError, match="altered"):
        owner.evaluate_relational_seeded_response(declaration, study=study)


def test_unknown_study_is_not_coerced_to_a_supported_protocol():
    with pytest.raises(ValueError, match="unsupported"):
        owner.prepare_relational_seeded_response(study="extended")


@pytest.mark.parametrize("horizon", (0, -1, 5, True, 1.125, "9/8"))
def test_budget_horizon_requires_bounded_exact_input(horizon):
    with pytest.raises((TypeError, ValueError)):
        owner.prepare_relational_seeded_response(study="target-budget", horizon=horizon)


def test_short_protocol_cannot_be_extended():
    with pytest.raises(ValueError, match="fixed horizon"):
        owner.prepare_relational_seeded_response(horizon=Q(1, 8))


def test_declared_continuation_changes_only_horizon_and_binds_evaluation(monkeypatch):
    original = owner.prepare_relational_seeded_response(study="target-budget")
    continuation = owner.prepare_relational_seeded_response(
        study="target-budget", horizon=Q(9, 8)
    )
    assert {key for key in original if original[key] != continuation[key]} == {
        "horizon"
    }
    calls = []

    class StopBeforeFlow(Exception):
        pass

    def stopped(initial, **kwargs):
        calls.append(kwargs)
        raise StopBeforeFlow

    monkeypatch.setattr(owner, "certify_relational_reflected_transit", stopped)
    with pytest.raises(ValueError, match="altered"):
        owner.evaluate_relational_seeded_response(continuation, study="target-budget")
    assert calls == []
    with pytest.raises(StopBeforeFlow):
        owner.evaluate_relational_seeded_response(
            continuation, study="target-budget", horizon=Q(9, 8)
        )
    assert len(calls) == 1
    assert calls[0]["horizon"] == Q(9, 8)
    assert calls[0]["time_step"] == Q(1, 64)


def test_earlier_certified_exclusion_survives_later_unresolved_endpoint():
    declaration = owner.prepare_relational_seeded_response(study="target-budget")
    report = SimpleNamespace(
        target_budget=I(-1, 1),
        admitted=False,
        validated_horizon=Q(3, 4),
        observations=(
            SimpleNamespace(time=Q(1, 4), target_budget=I(1, 2)),
            SimpleNamespace(time=Q(1, 2), target_budget=I(-2, -1)),
            SimpleNamespace(time=Q(3, 4), target_budget=I(-1, 1)),
        ),
        to_dict=lambda: {"fixture": "independently supplied enclosures"},
    )
    response = owner._budget_response(report, Q(0), I(0), declaration)
    assert response["target_budget_verdict"] == "unavailable_horizon"
    assert response["budget_sign_at_validated_horizon"] == "undecided"
    assert not response["passed"]
    assert response["future_target_excluded_from_validated_horizon"]
    assert Q(**response["first_certified_target_exclusion_time"]) == Q(1, 2)
