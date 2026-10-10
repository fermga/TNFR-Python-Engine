"""Synthetic checks of the frozen receiver runner, without its source trajectory."""

from dataclasses import dataclass
from fractions import Fraction as Q

import pytest

from tnfr.mathematics._comparison_flow import scalar_affine_upper
from tnfr.mathematics._rational_interval import I
from tnfr.research import organized_receiver_enclosure as owner


def test_scalar_comparison_uses_shared_arithmetic():
    assert owner._discrepancy_upper is scalar_affine_upper


def test_static_zero_reference_carries_full_error_and_signed_contraction():
    """Use a stationary state unrelated to the original two-ramp source."""
    incoming = Q(1, 1000)
    result = owner._advance_receiver_reference(
        (Q(0),) * 13, incoming, time=Q(0), form_bound=Q(0)
    )
    assert result["accepted"]
    assert result["osl"]["growth_rate_upper_bound"] < 0
    assert result["strict_error_margin"] > 0
    assert Q(4, 5) * incoming < result["endpoint_phase_error"] < incoming
    assert max(abs(value) for value in result["reference_step"]["endpoint_center"]) < Q(
        1, 10**25
    )
    assert result["endpoint"]["windings"] == (0, 0, 0)
    assert not result["endpoint"]["capture_inequalities_pass"]


@dataclass(frozen=True)
class _SyntheticStep:
    tube: tuple = (I(0),) * 13
    endpoint_center: tuple = (Q(0),) * 13
    endpoint_radius: Q = Q(1, 100)


@dataclass(frozen=True)
class _SyntheticOSL:
    growth_rate_upper_bound: Q = Q(0)
    certified: bool = True


@dataclass(frozen=True)
class _SyntheticObservation:
    donor_registered_branch_margins: tuple = (Q(1), Q(1))
    windings: tuple = (1, 1, 0)
    capture_inequalities_pass: bool = False


@dataclass(frozen=True)
class _SyntheticSource:
    status: str = "certified_uniform_slow_admission"
    release_phase_error_norm_upper_bound: Q = Q(1, 1000)
    release_form_norm_upper_bound: Q = Q(1, 100)
    donor_release_bridge_certified: bool = True


def test_step_adds_lifted_recentering_error_once_without_reset(monkeypatch):
    monkeypatch.setattr(
        owner,
        "validated_metric_taylor_step",
        lambda *a, **kw: (_SyntheticStep(), None, None),
    )
    monkeypatch.setattr(owner, "_receiver_local_osl", lambda *a, **kw: _SyntheticOSL())
    monkeypatch.setattr(
        owner, "_observe_receiver_tube", lambda *a, **kw: _SyntheticObservation()
    )
    radius = Q(1, 1000)
    result = owner._advance_receiver_reference(
        (Q(0),) * 13, radius, time=Q(0), form_bound=Q(0)
    )
    expected = radius + Q(11, 10**6) * Q(1, 8) + Q(1, 100)
    assert result["accepted"]
    assert expected <= result["endpoint_phase_error"] < expected + Q(1, 10**30)
    assert result["whole_step_phase_error"] < Q(1, 100)


def test_reference_validation_failure_does_not_consume_an_osl(monkeypatch):
    monkeypatch.setattr(
        owner,
        "validated_metric_taylor_step",
        lambda *a, **kw: (None, (I(0),) * 13, "synthetic_failure"),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("failed reference cannot reach growth certification")

    monkeypatch.setattr(owner, "_receiver_local_osl", forbidden)
    result = owner._advance_receiver_reference(
        (Q(0),) * 13, Q(1, 1000), time=Q(0), form_bound=Q(0)
    )
    assert not result["accepted"]
    assert result["reason"] == "synthetic_failure"


@pytest.fixture
def synthetic_source(monkeypatch):
    """Isolate orchestration from scientific source and flow evaluation."""
    monkeypatch.setattr(owner, "receiver_enclosure_source", _SyntheticSource)
    monkeypatch.setattr(
        owner, "_observe_receiver_tube", lambda *a, **kw: _SyntheticObservation()
    )


def _record(
    *, margins=(Q(1), Q(1)), windings=(1, 1, 0), capture=False, error=Q(1, 1000)
):
    return {
        "accepted": True,
        "reference_step": {"endpoint_center": (Q(0),) * 13},
        "endpoint_phase_error": error,
        "donor_corridor": {"donor_registered_branch_margins": margins},
        "endpoint": {"windings": windings, "capture_inequalities_pass": capture},
    }


def test_unavailable_donor_margin_continues_to_robust_loss(
    monkeypatch, synthetic_source
):
    responses = iter(
        [
            _record(margins=(Q(-1), Q(1)), windings=(None, 1, 0), capture=True),
            _record(windings=(0, 1, 0), error=Q(3, 5)),
        ]
    )
    monkeypatch.setattr(
        owner, "_advance_receiver_reference", lambda *a, **kw: next(responses)
    )
    retained = []
    result = owner.run_receiver_enclosure(record_step=retained.append)
    assert result["status"] == "certified_donor_loss"
    assert result["accepted_steps"] == 2
    assert len(retained) == 2
    assert not result["donor_history_certified"]
    assert result["first_unavailable_donor_interval"] == (Q(0), Q(1, 8))
    assert result["elapsed_slow_time"] == Q(1, 4)


def test_capture_requires_whole_donor_history(monkeypatch, synthetic_source):
    responses = iter(
        [
            _record(margins=(Q(0), Q(1)), windings=(1, 1, 1), capture=True),
            {"accepted": False, "reason": "synthetic_later_failure"},
        ]
    )
    monkeypatch.setattr(
        owner, "_advance_receiver_reference", lambda *a, **kw: next(responses)
    )
    retained = []
    result = owner.run_receiver_enclosure(record_step=retained.append)
    assert result["status"] == "unavailable"
    assert result["reason"] == "synthetic_later_failure"
    assert result["accepted_steps"] == 1
    assert result["final_observation"]["capture_inequalities_pass"]
    assert len(retained) == 2


def test_joint_capture_with_history_stops_at_first_certified_endpoint(
    monkeypatch, synthetic_source
):
    monkeypatch.setattr(
        owner,
        "_advance_receiver_reference",
        lambda *a, **kw: _record(windings=(1, 1, 1), capture=True),
    )
    result = owner.run_receiver_enclosure(record_step=lambda item: None)
    assert result["status"] == "certified_joined_capture_with_donor_retention"
    assert result["accepted_steps"] == 1
    assert result["donor_history_certified"]


def test_error_budget_is_unavailable_not_negative_evidence(
    monkeypatch, synthetic_source
):
    monkeypatch.setattr(
        owner, "_advance_receiver_reference", lambda *a, **kw: _record(error=Q(3, 5))
    )
    result = owner.run_receiver_enclosure(record_step=lambda item: None)
    assert result["status"] == "unavailable"
    assert result["reason"] == "phase_error_budget_exhausted"
    assert result["accepted_steps"] == 1


def test_arithmetic_failure_preserves_the_validated_prefix(
    monkeypatch, synthetic_source
):
    def eventually_unavailable(*args, time, **kwargs):
        if time == 0:
            return _record()
        raise ArithmeticError("synthetic OSL failure")

    monkeypatch.setattr(owner, "_advance_receiver_reference", eventually_unavailable)
    retained = []
    result = owner.run_receiver_enclosure(record_step=retained.append)
    assert result["status"] == "unavailable"
    assert result["reason"] == "numerical_or_domain_validation_error"
    assert result["accepted_steps"] == 1
    assert result["elapsed_slow_time"] == Q(1, 8)
    assert len(retained) == 2
    assert retained[-1]["error_type"] == "ArithmeticError"
    assert retained[-1]["error"] == "synthetic OSL failure"


def test_fixed_horizon_does_not_extend_or_retry(monkeypatch, synthetic_source):
    times = []

    def unchanged(*args, time, **kwargs):
        times.append(time)
        return _record()

    monkeypatch.setattr(owner, "_advance_receiver_reference", unchanged)
    retained = []
    result = owner.run_receiver_enclosure(record_step=retained.append)
    assert result["status"] == "unavailable"
    assert result["reason"] == "horizon_exhausted"
    assert result["accepted_steps"] == len(times) == len(retained) == 256
    assert times == [Q(index, 8) for index in range(256)]
    assert result["elapsed_slow_time"] == 32


def test_failed_source_admission_never_evaluates_a_reference(monkeypatch):
    monkeypatch.setattr(
        owner,
        "receiver_enclosure_source",
        lambda: _SyntheticSource(status="unavailable"),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("failed source cannot reach a numerical response")

    monkeypatch.setattr(owner, "_advance_receiver_reference", forbidden)
    monkeypatch.setattr(owner, "_observe_receiver_tube", forbidden)
    result = owner.run_receiver_enclosure(record_step=forbidden)
    assert result["status"] == "unavailable"
    assert result["reason"] == "source_admission_failed"
    assert result["accepted_steps"] == 0


def test_policy_is_fresh_and_does_not_depend_on_retained_results():
    modified = owner.receiver_enclosure_policy()
    modified["maximum_steps"] = 1
    assert owner.receiver_enclosure_policy()["maximum_steps"] == 256
