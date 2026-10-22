"""Actual-cell analytical admission, with no acquired response or event."""

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q
from pathlib import Path

import pytest

from tnfr.mathematics import _validated_taylor
from tnfr.mathematics._rational_interval import I
from tnfr.operators import network_stage, word_execution
from tnfr.research import maintained_child_joint_window
from tnfr.research import maintained_child_oz_timing as owner
from tnfr.research import maintained_sha_motion
from tnfr.research.maintained_parent_thol import _signature

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/assets/maintained_child_joint_window"


@pytest.fixture(scope="module", autouse=True)
def no_response_execution():
    def forbidden(*args, **kwargs):
        pytest.fail("analytical child admission cannot produce a response")

    with pytest.MonkeyPatch.context() as patch:
        for module, name in (
            (maintained_child_joint_window, "inspect_child_joint_window"),
            (maintained_sha_motion, "certify_sha_retention"),
            (maintained_sha_motion, "validated_box_taylor_step"),
            (_validated_taylor, "validated_box_taylor_step"),
            (network_stage, "execute_dissonance_stage"),
            (network_stage, "execute_self_organization_stage"),
            (word_execution, "execute_network_operator_stage"),
        ):
            patch.setattr(module, name, forbidden)
        yield


@pytest.fixture(scope="module")
def admission():
    original = owner.load_child_joint_window
    captured = []

    def load(*args, **kwargs):
        source = original(*args, **kwargs)
        captured.append((source, _signature(source.retention.admitted.graph)))
        return source

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(owner, "load_child_joint_window", load)
        result = owner.admit_child_oz_timing(EVIDENCE, repository_root=ROOT)
    assert len(captured) == 1
    source, before = captured[0]
    yield result, source
    assert _signature(source.retention.admitted.graph) == before


def test_fixed_absolute_clock_and_shared_sample_error_budget(admission):
    result, _ = admission
    assert result.source_time == Q(3670017, 524288)
    assert result.elapsed_sample_times == (Q(3, 128), Q(25, 1024), Q(13, 512))
    expected = tuple(Q(value, 524288) for value in (3682305, 3682817, 3683329))
    assert (
        result.reader_budget.sample_times
        == result.sample_budget.sample_times
        == expected
    )
    for time in expected:
        assert Q.from_float(float(time)) == time
    # Four value-error coefficients divided by h^2, independently evaluated.
    assert 4 * Q(1, 2**24) / Q(1, 1024) ** 2 == Q(1, 4)
    assert result.reader_budget.acceleration_sample_error_bound == Q(1, 4)
    assert result.sample_budget.acceleration_sample_error_bound == Q(1, 4)


def test_whole_window_bound_and_reader_exactness_have_separate_premises(admission):
    result, source = admission
    reader = result.reader_budget
    assert reader.exact and reader.arithmetic_error_bound == 0
    assert reader.abstention_reason is None
    assert reader.represented_acceleration_bounds.lo > Q(443, 4)
    assert reader.represented_acceleration_bounds.lo > result.threshold
    assert source.bounds.form_rate.lo > 4 and source.bounds.form_rate.hi < 5
    for secant in reader.represented_secant_bounds:
        assert secant.lo > 4 - Q(1, 8192)
        assert secant.hi < 5 + Q(1, 8192)
    assert result.sample_budget.acceleration_truncation_error_bound > 0
    assert reader.represented_acceleration_bounds.lo == (
        source.bounds.form_acceleration.lo - Q(1, 4)
    )


def test_future_oz_form_admits_while_actual_source_does_not(admission):
    result, _ = admission
    assert not result.source_form_admits_strict_oz
    assert result.future_oz_scalar_admission
    assert result.represented_form_bounds.lo > Q.from_float(0.2)
    assert result.incoming_pressure_bounds.lo > Q(34, 1000)
    assert result.incoming_pressure_bounds.hi < Q(37, 1000)
    assert result.minimum_principal_margin > Q(158, 100)


def test_raw_oz_is_nonzero_but_enabled_propagation_is_below_cutoff(admission):
    result, source = admission
    weight = source.retention.prediction.conductances[
        source.retention.prediction.edges.index((2, 20))
    ]
    assert result.gain == Q.from_float(1.3183098861837907)
    assert result.local_increment_bounds.lo > 0
    # The interval bound includes the shared incoming pressure, rather than
    # subtracting unrelated raw/incoming extremes and losing their correlation.
    ideal_maximum = (result.gain - 1) * result.incoming_pressure_bounds.hi * weight
    assert result.propagation_increment_upper_bound >= ideal_maximum
    assert result.propagation_increment_upper_bound < Q(1, 100)
    assert result.propagation_cutoff == Q.from_float(0.05)
    assert result.propagation_enabled and result.propagation_cutoff_excludes_parent
    assert result.conditional_storage_jump == 0


def test_a_planned_word_supplies_no_executed_prefix_or_child_samples(admission):
    result, source = admission
    assert result.source_oz_grammar_allowed
    assert "U4b" in result.source_thol_error_rules
    assert result.selected_word == ("dissonance", "self_organization", "silence")
    assert not result.executed_event
    assert not result.acquired_history
    assert not result.acquired_u4b_context
    assert not result.full_thol_admission_established
    graph = source.retention.admitted.graph
    assert tuple(graph.nodes[owner.CHILD]["glyph_history"]) == ()
    assert "_epi_history" not in graph.nodes[owner.CHILD]


@pytest.mark.parametrize(
    "changes",
    (
        {"OZ_NOISE_MODE": True},
        {"OZ_ENABLE_PROPAGATION": False},
        {"OZ_MIN_EPI": 0.3},
        {"OZ_MAX_DNFR": 0.01},
        {"OZ_MIN_VF": 200.0},
        {"VALIDATE_OPERATOR_PRECONDITIONS": False},
        {"GRAMMAR_REJECTION_MODE": "substitute"},
    ),
)
def test_actual_policy_changes_cannot_inherit_the_admission(
    admission, monkeypatch, changes
):
    _, source = admission
    altered = deepcopy(source)
    altered.retention.admitted.graph.graph.update(changes)
    monkeypatch.setattr(owner, "load_child_joint_window", lambda *a, **k: altered)
    with pytest.raises(ValueError):
        owner.admit_child_oz_timing(EVIDENCE, repository_root=ROOT)


def test_unresolved_arithmetic_cannot_be_reported_as_measured_readiness(
    admission, monkeypatch
):
    _, source = admission
    altered = replace(source, bounds=replace(source.bounds, form_rate=I(-1, 5)))
    monkeypatch.setattr(owner, "load_child_joint_window", lambda *a, **k: altered)
    with pytest.raises(ValueError, match="not admitted"):
        owner.admit_child_oz_timing(EVIDENCE, repository_root=ROOT)
