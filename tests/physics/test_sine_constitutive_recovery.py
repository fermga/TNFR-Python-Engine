"""Shared isolated recovery is independent of any later contact consumer."""

from dataclasses import replace
from fractions import Fraction as Q

import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.physics import _sine_constitutive_handoff as contact_owner
from tnfr.physics import _sine_constitutive_recovery as owner

ERROR = Q(1, 10**10)


@pytest.fixture(scope="module", autouse=True)
def no_scientific_execution():
    with forbid_sine_regeneration():
        yield


def _assess(**changes):
    return owner._assess_constitutive_recovery(
        **(
            dict(deformation=Q(1), form_error_bound=ERROR, phase_error_bound=ERROR)
            | changes
        )
    )


@pytest.fixture(scope="module")
def recovery():
    return _assess()


def test_isolated_recovery_has_no_contact_premises(recovery, monkeypatch):
    # A zero barrier rejects all contact branches without changing the
    # isolated source or the valid recovery subsequently used elsewhere.
    monkeypatch.setattr(contact_owner, "_JOINED_BARRIER", Q(0))
    contact = contact_owner._assess_constitutive_handoff(
        deformation=1, form_error_bound=ERROR, phase_error_bound=ERROR
    )
    assert recovery.status == "certified_constitutive_recovery"
    assert recovery.recovery_certified and not recovery.unresolved_conditions
    assert contact.status == "unavailable"
    assert contact.handoff_certified
    assert all(not branch.identity_certified for branch in contact.branches)
    assert recovery.source == contact.source
    assert (
        recovery.returned_lyapunov_upper_bound == contact.returned_lyapunov_upper_bound
    )
    assert recovery.scaled_handoff_time == 100 + 10**13
    assert recovery.endpoint_radius == Q(1, 10**70)
    assert not hasattr(recovery, "attachment_work_bounds")
    assert not hasattr(recovery, "branches")


def test_each_call_rebuilds_source_and_only_class_one_is_consumed(monkeypatch):
    actual = owner._assess_constitutive_class_pair
    calls = []

    def class_two_not_admitted(**kwargs):
        calls.append(kwargs)
        source = actual(**kwargs)
        return replace(
            source,
            formation_certified_by_class=(
                source.formation_certified_by_class[0],
                False,
            ),
            source_budget_certified_by_class=(
                source.source_budget_certified_by_class[0],
                False,
            ),
            status="unavailable",
        )

    monkeypatch.setattr(
        owner, "_assess_constitutive_class_pair", class_two_not_admitted
    )
    for coefficient in (Q(0), Q(1)):
        report = _assess(deformation=coefficient)
        assert report.recovery_certified
        assert report.source.status == "unavailable"
        assert report.endpoint_form_norm_squared_upper_bound < Q(1, 10**140)
        assert report.endpoint_phase_norm_squared_upper_bound < Q(1, 10**140)
    assert len(calls) == 2
    assert [call["deformation"] for call in calls] == [0, 1]
    assert all(
        call["scaled_time"] == 100 and call["radius"] == Q(1, 12) for call in calls
    )


def test_nonpositive_recovery_margin_is_unavailable(monkeypatch):
    monkeypatch.setattr(owner, "_ENDPOINT_RADIUS", Q(0))
    report = _assess()
    assert report.source_certified
    assert not report.recovery_certified
    assert report.endpoint_form_margin < 0 and report.endpoint_phase_margin < 0
    assert "actual_form_handoff_radius_not_certified" in report.unresolved_conditions
    assert "actual_phase_handoff_radius_not_certified" in report.unresolved_conditions


@pytest.mark.parametrize(
    "field,value",
    [
        ("deformation", True),
        ("deformation", Q(-1)),
        ("form_error_bound", float("nan")),
        ("phase_error_bound", True),
        ("phase_error_bound", Q(-1)),
    ],
)
def test_bad_raw_values_reject_before_source_admission(monkeypatch, field, value):
    def forbidden(**kwargs):
        pytest.fail("source construction preceded raw scalar admission")

    monkeypatch.setattr(owner, "_assess_constitutive_class_pair", forbidden)
    with pytest.raises((TypeError, ValueError)):
        _assess(**{field: value})


@pytest.mark.parametrize(
    "extra",
    [{"source": None}, {"relaxation_duration": Q(0)}, {"endpoint_radius": Q(0)}],
)
def test_reports_and_altered_recovery_protocol_are_not_admitted(extra):
    with pytest.raises(TypeError):
        _assess(**extra)
