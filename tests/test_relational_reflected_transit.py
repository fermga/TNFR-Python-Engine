"""Continuous-proof admission controls; no reserved source/receiver response."""

from fractions import Fraction as Q

import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.mathematics._rational_interval import I, pi_interval
from tnfr.physics import relational_reflected_transit as owner

MODEL = RelationalExchangeModel(1, phase_domain="regular")


@pytest.fixture(scope="module")
def consensus():
    # This separate exact equilibrium is not the reserved twisted preparation.
    return owner.certify_relational_reflected_transit(
        (0,) * 8, model=MODEL, horizon=Q(1, 64), time_step=Q(1, 128), order=3
    )


def test_continuous_consensus_is_enclosed_without_a_formation_claim(consensus):
    assert consensus.admitted and consensus.validated_horizon == Q(1, 64)
    assert len(consensus.steps) == len(consensus.observations) == 2
    assert all(value.contains(0) for value in consensus.endpoint)
    assert consensus.endpoint_storage.contains(0)
    assert consensus.cumulative_loss.contains(0)
    assert consensus.receiver_winding_zero_throughout
    assert not consensus.source_winding_one_throughout
    assert consensus.target_budget.hi < 0  # Admitted flow is not target capture.
    for step in consensus.steps:
        assert step.picard_interior_margin > 0
        assert len(step.domain_lower_bounds) == 10
        assert min(step.domain_lower_bounds) > 0
        assert all(
            value.subset_of(tube) for value, tube in zip(step.endpoint, step.tube)
        )


def test_certificate_projection_keeps_exact_loss_and_independent_flags(consensus):
    data = consensus.to_dict()["report"]
    assert data["status"] == "admitted"
    assert data["source_winding_one_throughout"] is False
    assert data["receiver_winding_zero_throughout"] is True
    loss = data["cumulative_loss"]
    assert Q(**loss["lo"]) == consensus.cumulative_loss.lo
    assert Q(**loss["hi"]) == consensus.cumulative_loss.hi


@pytest.mark.parametrize(
    "horizon,step,order",
    (
        (True, Q(1, 8), 3),
        (0, 1, 3),
        (1, 0, 3),
        (257, 1, 3),
        (1, 1, True),
        (1, 1, 17),
        (0.125, 1, 3),
    ),
)
def test_invalid_numerical_declarations_reject(horizon, step, order):
    with pytest.raises((ValueError, TypeError)):
        owner.certify_relational_reflected_transit(
            (0,) * 8, model=MODEL, horizon=horizon, time_step=step, order=order
        )


def test_model_and_initial_component_are_not_repaired():
    with pytest.raises(ValueError, match="regular"):
        owner.certify_relational_reflected_transit(
            (0,) * 8, model=RelationalExchangeModel(1), horizon=1, time_step=1
        )
    bad = (I(0),) * 5 + (pi_interval(), I(0), I(0))
    with pytest.raises(ValueError):
        owner.certify_relational_reflected_transit(
            bad, model=MODEL, horizon=1, time_step=1
        )


def test_first_unresolved_tube_is_retained_without_retry(monkeypatch):
    calls = []
    failed = (I(-1, 1),) * 8

    def stop(*args, **kwargs):
        calls.append((args, kwargs))
        return None, failed, "synthetic_inconclusive_domain_bound"

    monkeypatch.setattr(owner, "validated_taylor_step", stop)
    result = owner.certify_relational_reflected_transit(
        (0,) * 8, model=MODEL, horizon=Q(1, 8), time_step=Q(1, 64)
    )
    assert len(calls) == 1
    assert result.status == "unavailable" and result.validated_horizon == 0
    assert result.failed_tube == failed
    assert result.steps == result.observations == ()
    assert result.unavailable_reasons == (
        "synthetic_inconclusive_domain_bound",
        "requested_horizon_not_validated",
    )


def test_storage_intersection_cannot_hide_an_inconsistent_certificate():
    with pytest.raises(ArithmeticError, match="disjoint"):
        owner._intersection(I(0, 1), I(2, 3))


def test_altered_reserved_protocol_rejects_before_any_response(monkeypatch):
    from tnfr.research import relational_seeded_response as response

    def forbidden(*args, **kwargs):
        pytest.fail("an altered protocol must not evolve")

    monkeypatch.setattr(response, "certify_relational_reflected_transit", forbidden)
    declaration = response.prepare_relational_seeded_response()
    declaration["order"] += 1
    with pytest.raises(ValueError, match="altered"):
        response.evaluate_relational_seeded_response(declaration)
