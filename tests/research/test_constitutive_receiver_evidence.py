"""Read-only full-source law comparison and primitive evidence boundaries."""

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.sine_evidence_helpers import forbid_sine_regeneration
from tnfr.research import constitutive_receiver_comparison as owner

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", autouse=True)
def no_trajectory_or_unpacked_evidence():
    original_open = Path.open

    def archived_only(path, *args, **kwargs):
        resolved = path.resolve()
        if resolved == ROOT / (owner._STEM + ".json") or resolved.is_relative_to(
            ROOT / (owner._STEM + ".steps")
        ):
            pytest.fail("comparison requires only the committed response archive")
        return original_open(path, *args, **kwargs)

    with forbid_sine_regeneration(), pytest.MonkeyPatch.context() as patch:
        patch.setattr(Path, "open", archived_only)
        yield


@pytest.fixture(scope="module")
def retained():
    return owner._read_reference(ROOT)


@pytest.fixture(scope="module")
def comparison(retained):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(owner, "_read_reference", lambda root: retained)
        return owner.compare_constitutive_receiver_prefix(ROOT)


def test_transport_change_rejects_before_json_or_scientific_admission(monkeypatch):
    monkeypatch.setattr(owner, "read_bytes_bounded", lambda *a, **kw: b"altered")

    def forbidden(*args, **kwargs):
        pytest.fail("altered transport reached parsing")

    monkeypatch.setattr(owner, "json_loads", forbidden)
    with pytest.raises(ValueError, match="transport differs"):
        owner._read_reference(ROOT)


def test_every_retained_step_rebuilds_endpoint_and_carries_reference_center(retained):
    center = retained[0]["reference_step"]["initial_center"]
    for index, record in enumerate(retained, 1):
        step = owner._admit_reference_step(record, index=index, initial_center=center)
        assert step.time == Q(index - 1, 8)
        assert step.duration == Q(1, 8)
        assert step.endpoint_center == tuple(
            x.midpoint for x in step.reference_endpoint
        )
        center = step.endpoint_center
    assert len(retained) == 76 and step.time + step.duration == Q(19, 2)


@pytest.mark.parametrize(
    "field,value,message",
    (
        ("time", False, "exact record"),
        ("duration", Q(1, 16), "time or duration"),
        ("order", True, "order"),
        ("initial_radius", Q(1, 1000), "initial radius"),
        ("picard_interior_margin", Q(0), "Picard premise"),
        ("domain_lower_bounds", (False,), "exact record"),
        ("initial_center", (Q(0),) * 13, "center seam"),
        ("endpoint_center", (Q(0),) * 13, "recentering"),
        ("metric", ((True,) * 13,) * 13, "exact record"),
        ("tube", ({"lo": Q(2), "hi": Q(1)},) * 13, "reversed interval"),
        (
            "center_solution_endpoint",
            ({"lo": Q(0), "hi": Q(0)},) * 13,
            "endpoint reconstruction",
        ),
    ),
)
def test_consumed_primitive_or_derived_endpoint_cannot_be_forged(
    retained, field, value, message
):
    record = deepcopy(retained[0])
    center = record["reference_step"]["initial_center"]
    record["reference_step"][field] = value
    with pytest.raises((TypeError, ValueError), match=message):
        owner._admit_reference_step(record, index=1, initial_center=center)


def test_cached_old_law_growth_winding_and_radius_are_not_new_premises(retained):
    record = deepcopy(retained[0])
    center = record["reference_step"]["initial_center"]
    expected = owner._admit_reference_step(record, index=1, initial_center=center)
    for key in (
        "accepted",
        "osl",
        "initial_phase_error",
        "endpoint_phase_error",
        "donor_corridor",
        "endpoint",
        "donor_history_certified",
    ):
        record[key] = "not consumed"
    for key in (
        "endpoint_radius",
        "local_metric_error_upper_bound",
        "local_coordinate_error_bounds",
        "growth_rate_upper_bound",
    ):
        record["reference_step"][key] = "not consumed"
    assert (
        owner._admit_reference_step(record, index=1, initial_center=center) == expected
    )


def test_source_budget_must_be_fresh_and_pass_before_reading_archive(monkeypatch):
    source = owner._assess_constitutive_receiver_source(
        deformation_bound=Q(1, 10000),
        form_error_bound=Q(1, 10**10),
        phase_error_bound=Q(1, 10**10),
    )
    monkeypatch.setattr(
        owner,
        "_assess_constitutive_receiver_source",
        lambda **kw: replace(source, uniform_velocity_defect_upper_bound=Q(1)),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("unavailable own-law source reached reference comparison")

    monkeypatch.setattr(owner, "_read_reference", forbidden)
    with pytest.raises(ValueError, match="velocity budget"):
        owner.compare_constitutive_receiver_prefix(ROOT)


@pytest.mark.parametrize(
    "flags,failed_step,donors,expected_receiver,expected_donors,status",
    (
        ((None, False), None, None, False, None, "comparison_outcome_unavailable"),
        ((False,), 2, None, False, None, "comparison_outcome_unavailable"),
        ((None,), 2, None, None, None, "comparison_outcome_unavailable"),
        (
            (True,),
            None,
            False,
            True,
            False,
            "certified_uniform_receiver_zero_joint_donor_loss_excluded",
        ),
    ),
)
def test_report_wiring_retains_counterexamples_and_unavailable_evidence(
    retained,
    monkeypatch,
    flags,
    failed_step,
    donors,
    expected_receiver,
    expected_donors,
    status,
):
    from tnfr.physics import _constitutive_receiver_comparison as kernel

    count = 0

    def synthetic_step(**kwargs):
        nonlocal count
        count += 1
        return SimpleNamespace(
            status=(
                "unavailable" if count == failed_step else "certified_local_comparison"
            ),
            endpoint_error=Q(1, 100),
            receiver_zero_throughout=flags[min(count - 1, len(flags) - 1)],
            donor_endpoint_nonunit=donors,
        )

    monkeypatch.setattr(kernel, "_compare_constitutive_receiver_step", synthetic_step)
    monkeypatch.setattr(owner, "_read_reference", lambda root: retained)
    result = owner.compare_constitutive_receiver_prefix(ROOT)
    assert result["receiver_zero_throughout"] is expected_receiver
    assert result["donor_loss_at_endpoint"] is expected_donors
    assert result["status"] == status
    assert result["complete_prefix_certified"] is (failed_step is None)
    if failed_step is not None:
        assert result["endpoint_phase_error"] is None
        assert len(result["steps"]) == failed_step


def test_full_law_interval_uses_original_source_and_all_preserved_steps(comparison):
    policy, source = comparison["policy"], comparison["source"]
    assert policy["deformation_bound"] == Q(1, 10000)
    assert source.deformation_interval == (-Q(1, 10000), Q(1, 10000))
    assert source.relative_preparation_dimension == 48
    assert source.release_windings == (1, 1, 0)
    assert policy["form_error_bound"] == policy["phase_error_bound"] == Q(1, 10**10)
    assert policy["elapsed_slow_prefix"] == Q(19, 2)
    assert policy["original_prospective_maximum_horizon"] == 32
    assert comparison["complete_prefix_certified"]
    assert len(comparison["steps"]) == 76
    assert comparison["trajectory_evaluated"] is False
    incoming = policy["initial_phase_error_budget"]
    for step in comparison["steps"]:
        assert step.strict_error_margin > 0
        assert step.cones[0].initial_error == incoming
        assert len(step.cones) == 8
        assert step.osl.phase_mean_leaf_dimension == 26
        incoming = step.endpoint_error
    assert incoming == comparison["endpoint_phase_error"]


def test_receiver_exclusion_and_donor_unavailability_remain_separate(comparison):
    assert (
        comparison["status"]
        == "certified_uniform_receiver_zero_donor_outcome_unavailable"
    )
    assert comparison["receiver_zero_throughout"] is True
    assert comparison["donor_loss_at_endpoint"] is None
    assert Q(286, 1000) < comparison["endpoint_phase_error"] < Q(287, 1000)
    receiver_margin = min(
        cone.zero_branch_margins[2]
        for step in comparison["steps"]
        for cone in step.cones
    )
    assert Q(108, 100) < receiver_margin < Q(109, 100)
    endpoint = comparison["steps"][-1]
    assert endpoint.endpoint_windings == (None, None, 0)
    assert all(
        margin < -Q(349, 1000) for margin in endpoint.endpoint_zero_branch_margins[:2]
    )
