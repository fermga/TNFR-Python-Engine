"""Actual child history compatibility, independent of an acquired response."""

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q
from pathlib import Path

import pytest

from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_VF
from tnfr.mathematics import _validated_taylor
from tnfr.mathematics._rational_interval import I, cos, pi_interval, sin
from tnfr.operators import network_stage, word_execution
from tnfr.research import local_capacity_history
from tnfr.research import maintained_child_history as owner
from tnfr.research import (
    maintained_parent_sha_execution,
    maintained_parent_um_execution,
    maintained_sha_motion_source,
)
from tnfr.research.maintained_parent_thol import _signature
from tnfr.research.maintained_sha_motion_source import load_sha_successor

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module", autouse=True)
def no_events_trajectories_or_pressure_refresh():
    def forbidden(*args, **kwargs):
        pytest.fail("analytical history admission cannot execute or acquire a response")

    with pytest.MonkeyPatch.context() as patch:
        for module, name in (
            (_validated_taylor, "validated_box_taylor_step"),
            (local_capacity_history, "rk4_coupled_step"),
            (local_capacity_history, "refresh_pressure"),
            (maintained_parent_sha_execution, "evaluate_sha_event"),
            (maintained_parent_sha_execution, "execute_network_operator_stage"),
            (maintained_parent_sha_execution, "refresh_pressure"),
            (maintained_parent_um_execution, "evaluate_um_event"),
            (maintained_parent_um_execution, "execute_network_operator_stage"),
            (maintained_parent_um_execution, "refresh_pressure"),
            (word_execution, "execute_network_operator_stage"),
            (network_stage, "execute_coupling_stage"),
            (network_stage, "execute_self_organization_stage"),
        ):
            patch.setattr(module, name, forbidden)
        yield


@pytest.fixture(scope="module")
def prepared():
    admitted = load_sha_successor(
        ROOT / "docs/assets/maintained_parent_sha", repository_root=ROOT
    )
    before = _signature(admitted.graph)
    result = owner.predict_child_history(admitted)
    yield admitted, result
    assert _signature(admitted.graph) == before


def _tight_overlap(left, right):
    assert max(left.lo, right.lo) <= min(left.hi, right.hi)
    assert abs(left.lo - right.lo) < Q(1, 2**80)
    assert abs(left.hi - right.hi) < Q(1, 2**80)


def test_actual_complete_source_and_child_owned_clock_are_retained(prepared):
    admitted, result = prepared
    motion = result.window.motion
    assert result.nodes == tuple(range(20)) + ("2_sub_0",)
    assert len(result.edges) == len(result.shifts) == 53
    assert len(motion.epi) == len(motion.phases) == 21
    assert result.window.target_index == result.nodes.index(result.child)
    assert motion.epi == tuple(map(I, admitted.prediction.epi))
    assert motion.phases == tuple(map(I, admitted.prediction.phases))
    assert motion.capacities == tuple(
        map(I, admitted.prediction.represented_capacity_after)
    )
    assert motion.capacities[2] != I(admitted.prediction.capacity_before[2])
    assert result.source_time == Q(admitted.graph.graph["_t"])
    assert result.elapsed_sample_times == (Q(0), Q(1, 2**16), Q(1, 2**15))
    assert result.sample_times == tuple(
        result.source_time + offset for offset in result.elapsed_sample_times
    )
    assert result.window.horizon == result.elapsed_sample_times[-1]
    assert result.source_child_hierarchy_level == 1
    assert admitted.graph.nodes[result.child]["parent_node"] == result.parent


def test_degree_one_child_equations_independently_give_the_true_window(prepared):
    _, result = prepared
    window, motion = result.window, result.window.motion
    edge = result.edges.index((result.parent, window.target_index))
    target = window.target_index
    assert motion.degrees[target] == 1
    delta = motion.phases[target] - motion.phases[result.parent]
    velocity = motion.phase_rate[target] - motion.phase_rate[result.parent]
    acceleration_bound = motion.edge_acceleration_magnitudes[edge].hi
    duration = window.horizon
    difference = (
        delta
        + I(0, duration) * velocity
        + I(
            -acceleration_bound * duration**2 / 2,
            acceleration_bound * duration**2 / 2,
        )
    )
    speed = velocity + I(-acceleration_bound * duration, acceleration_bound * duration)
    _tight_overlap(window.edge_phase_differences[edge], difference)
    _tight_overlap(window.edge_phase_velocities[edge], speed)
    _tight_overlap(window.pressure, -sin(difference) / pi_interval())
    _tight_overlap(
        window.form_acceleration,
        -motion.mobility[target] * cos(difference) * speed,
    )
    assert difference.hi < 0 and speed.hi < 0
    assert window.pressure.lo > Q(1, 20000)
    assert window.form_acceleration.lo > 202
    assert window.form_acceleration.hi < 211
    assert window.form.lo <= motion.epi[target].lo <= window.form.hi
    assert window.form.hi < Q(149, 1000) < result.minimum_form


def test_every_original_branch_has_a_fresh_strict_short_window_margin(prepared):
    _, result = prepared
    window = result.window
    assert len(window.retained_edge_indices) == 53
    assert window.retained_edge_indices == tuple(range(53))
    assert min(window.principal_margins) > Q(157, 100)
    for phase, margin, energy_margin in zip(
        window.edge_phase_differences,
        window.principal_margins,
        window.motion.horizon_remaining_margins,
        strict=True,
    ):
        assert margin == max(pi_interval().lo - phase.abs_max, energy_margin.lo)
    assert result.common_principal_identity_retained


def test_prospective_sample_errors_do_not_claim_an_observed_reader_result(prepared):
    _, result = prepared
    budget = result.sample_budget
    error = result.sample_error_allowance
    independent_error = (error + 2 * error + error) / result.sample_step**2
    assert error == Q(1, 2**30)
    assert independent_error == budget.acceleration_sample_error_bound == 16
    assert budget.third_derivative_bound == result.window.third_form_derivative_bound
    assert budget.third_derivative_bound > 0
    assert budget.acceleration_truncation_error_bound > 0
    assert result.prospective_exact_sample_acceleration_lower_bound == (
        result.window.form_acceleration.lo - independent_error
    )
    assert result.prospective_exact_sample_acceleration_lower_bound > 186
    assert result.temporal_capability
    assert not result.measured_history
    assert not result.reader_arithmetic_error_certified
    assert not result.actual_event


def test_own_live_history_form_and_grammar_remain_independent_obstructions(prepared):
    admitted, result = prepared
    observation = result.source_observation
    assert observation.history_length == 1
    assert observation.source == "epi_history"
    assert not observation.available and observation.value is None
    assert observation.reason == "insufficient_history"
    assert result.source_child_glyph_history == ()
    assert tuple(admitted.graph.nodes[result.parent]["glyph_history"]) == (
        "VAL",
        "THOL",
        "UM",
        "SHA",
    )
    assert not result.grammar_allowed
    assert result.grammar_error_rules == ("U4b",)
    assert result.blocked_u4b
    assert result.strict_precondition_error is not None
    assert "EPI too low" in result.strict_precondition_error
    assert result.minimum_form_excluded_whole_window
    assert not result.full_thol_readiness


def test_caller_cached_predictions_and_assessments_do_not_supply_premises(prepared):
    admitted, expected = prepared
    forged = replace(
        admitted,
        prediction=object(),
        assessment={"actual_fields": {"epi": (True,) * 21}},
        sequence=None,
        receipts=(),
    )
    assert owner.predict_child_history(forged) == expected


@pytest.mark.parametrize(
    "mutation",
    (
        lambda graph: set_attr(graph.nodes["2_sub_0"], ALIAS_EPI, 0.3),
        lambda graph: set_attr(graph.nodes["2_sub_0"], ALIAS_VF, False),
        lambda graph: graph.nodes["2_sub_0"].update(glyph_history=["VAL"]),
        lambda graph: graph.nodes["2_sub_0"].update(epi_time_history=[(7, 0.15)]),
        lambda graph: graph.graph.update(THOL_MIN_EPI=0.1),
    ),
)
def test_live_primitive_history_or_policy_changes_cannot_reuse_source(
    prepared, monkeypatch, mutation
):
    admitted, _ = prepared
    # The expensive source is already independently admitted by the module
    # fixture. Retain the real live signature check while isolating this
    # consumer's mutation wiring from redundant archive reconstruction.
    monkeypatch.setattr(
        maintained_sha_motion_source,
        "load_sha_successor",
        lambda *args, **kwargs: deepcopy(admitted),
    )
    graph = deepcopy(admitted.graph)
    mutation(graph)
    before = _signature(graph)
    with pytest.raises(ValueError, match="live SHA endpoint differs"):
        owner.predict_child_history(replace(admitted, graph=graph))
    assert _signature(graph) == before


def test_unassociated_source_rejects_without_claiming_history(prepared):
    admitted, _ = prepared
    with pytest.raises(ValueError, match="file-backed"):
        owner.predict_child_history(replace(admitted, evidence_dir=None))
