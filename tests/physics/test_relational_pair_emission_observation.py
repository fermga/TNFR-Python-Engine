"""Adversarial admission and orbit controls for the detached AL comparison."""

import copy
import json
import pickle
from dataclasses import dataclass, replace
from fractions import Fraction as Q

import mpmath as mp
import networkx as nx
import pytest

from tnfr.dynamics.relational import RelationalExchangeModel
from tnfr.errors.contextual import TNFRUserError
from tnfr.physics import relational_sine_scale as owner
from tnfr.sdk import export_to_json, relational_report_to_dict

MODEL = RelationalExchangeModel(1, epi_weight=0, phase_domain="regular")
ADMISSION_ERRORS = (TypeError, ValueError, TNFRUserError)


def _graph(*, form=(Q(1, 4), -Q(1, 4)), phase=(Q(1, 4), -Q(1, 4))):
    graph = nx.path_graph(2)
    for node in graph:
        graph.nodes[node].update(
            EPI=form[node], theta=phase[node], nu_f=1, delta_nfr="stale pressure"
        )
    graph.graph.update(GAMMA={"type": "none"}, EPI_MIN=-1, EPI_MAX=1, CLIP_MODE="hard")
    return graph


def _assess(graph=None, **changes):
    arguments = dict(
        reference_model=MODEL, pairs=((0, 1),), pair_index=0, boost=Q(1, 8)
    )
    arguments.update(changes)
    return owner.assess_sine_pair_emission(
        _graph() if graph is None else graph, **arguments
    )


def _snapshot(graph):
    return pickle.dumps(
        (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True))),
        protocol=5,
    )


def _mp(value):
    value = Q(value)
    return mp.mpf(value.numerator) / value.denominator


def _contains(bound, value):
    assert _mp(bound.lo) <= value <= _mp(bound.hi)


def test_equal_form_and_equal_phase_are_distinct_single_member_obstructions():
    """One missing signed coordinate is enough; storage equality is weaker."""
    phase_split = _assess(_graph(form=(0, 0)))
    assert not phase_split.single_member_source_orbit_equal
    assert phase_split.first_member.form_mean == Q(1, 16)
    assert phase_split.second_member.form_mean == Q(1, 16)
    assert phase_split.first_member.internal_form_squared == Q(1, 256)
    assert phase_split.second_member.internal_form_squared == Q(1, 256)
    with mp.workdps(100):
        correlation = mp.sin(mp.mpf(1) / 4) / 16
        _contains(phase_split.first_member.form_phase_correlation_bounds, correlation)
        _contains(phase_split.second_member.form_phase_correlation_bounds, -correlation)
    assert phase_split.first_member.form_phase_correlation_bounds.lo > 0
    assert phase_split.second_member.form_phase_correlation_bounds.hi < 0
    # On P2 phase is unchanged and both post-event form gaps have magnitude 1/8.
    assert (
        phase_split.first_member.form_pair[0] - phase_split.first_member.form_pair[1]
    ) ** 2 == (
        phase_split.second_member.form_pair[0] - phase_split.second_member.form_pair[1]
    ) ** 2

    form_split = _assess(_graph(phase=(0, 0)))
    assert not form_split.single_member_source_orbit_equal
    assert form_split.first_member.internal_form_squared == Q(25, 256)
    assert form_split.second_member.internal_form_squared == Q(9, 256)
    assert form_split.first_member.form_phase_correlation_bounds.lo == 0
    assert form_split.first_member.form_phase_correlation_bounds.hi == 0


def test_tip_has_one_unordered_outcome_without_selecting_a_fine_member():
    report = _assess(_graph(form=(0, 0), phase=(0, 0)))
    assert report.single_member_source_orbit_equal
    assert report.first_member.form_pair == (Q(1, 8), Q(0))
    assert report.second_member.form_pair == (Q(0), Q(1, 8))
    assert report.first_member.form_pair != report.second_member.form_pair
    assert report.first_member.form_mean == report.second_member.form_mean
    assert (
        report.first_member.internal_form_squared
        == report.second_member.internal_form_squared
    )
    assert report.whole_pair.form_pair == (Q(1, 8), Q(1, 8))
    assert report.whole_pair_structural_descent_certified
    assert not report.event_occurrence_derived
    assert not report.runtime_admission_certified


def test_clipping_can_collapse_whole_pair_but_not_unmarked_singleton():
    graph = _graph(form=(Q(3, 4), Q(7, 8)))
    report = _assess(graph, boost=Q(1, 2))
    assert report.member_proposals == (Q(1), Q(1))
    assert report.whole_pair.form_pair == (Q(1), Q(1))
    assert report.whole_pair.internal_form_squared == 0
    assert report.whole_pair.form_phase_correlation_bounds.lo == 0
    assert report.whole_pair.form_phase_correlation_bounds.hi == 0
    assert not report.single_member_source_orbit_equal
    assert report.whole_pair_structural_descent_certified

    # Different phases survive a complete no-op at the upper form boundary.
    noop = _assess(_graph(form=(1, 1)))
    assert noop.single_member_source_orbit_equal
    assert noop.first_member.form_pair == noop.second_member.form_pair == (Q(1), Q(1))
    # A collapsed configured interval is also an admitted pointwise map.
    collapsed_graph = _graph(form=(-Q(1, 4), -Q(1, 2)))
    collapsed_graph.graph.update(EPI_MIN=0, EPI_MAX=0)
    collapsed = _assess(collapsed_graph)
    assert collapsed.whole_pair.form_pair == (Q(0), Q(0))
    assert not collapsed.single_member_source_orbit_equal


def test_actual_soft_boundary_policy_is_shared_and_does_not_preserve_internals():
    from tnfr.operators.al_sha_stage_proposals import emission_epi_proposal

    graph = _graph(form=(Q(99, 100), Q(1, 4)))
    graph.graph.update(CLIP_MODE="soft", CLIP_SOFT_K="not consumed by AL")
    report = _assess(graph, boost=Q(1, 10000))
    expected = tuple(
        Q(emission_epi_proposal(graph.graph, graph.nodes[i]["EPI"], 0.0001)[1])
        for i in graph
    )
    assert report.member_proposals == expected
    assert report.whole_pair.form_pair == expected
    assert (
        report.whole_pair.internal_form_squared
        == ((expected[0] - expected[1]) / 2) ** 2
    )
    assert expected[0] > Q(float(Q(99, 100)) + 0.0001)
    assert report.whole_pair_structural_descent_certified


@pytest.mark.parametrize("mode, expected_mode", [("soft", "soft"), ("invalid", "hard")])
def test_effective_policy_matches_the_registered_glyph_without_reading_live_factors(
    mode, expected_mode
):
    from tnfr.config.defaults_core import CoreDefaults
    from tnfr.node import NodeNX
    from tnfr.operators import _op_AL
    from tnfr.types import require_finite_real_scalar_epi

    graph = _graph(form=(Q(99, 100), -Q(1, 4)))
    graph.graph.update(
        CLIP_MODE=mode,
        CLIP_SOFT_K="ignored by this actual AL policy",
        GLYPH_FACTORS={"AL_boost": "unconsumed invalid live factor"},
    )
    before = _snapshot(graph)
    report = _assess(graph, boost=Q(1, 10000))
    executed = copy.deepcopy(graph)
    for node in executed:
        _op_AL(NodeNX.from_graph(executed, node), {"AL_boost": 0.0001})
    assert report.member_proposals == tuple(
        Q(require_finite_real_scalar_epi(executed.nodes[i]["EPI"])) for i in executed
    )
    assert report.clip_policy == (
        Q(-1),
        Q(1),
        expected_mode,
        Q(CoreDefaults().CLIP_SOFT_K),
    )
    assert _snapshot(graph) == before


def test_exact_nonzero_phase_is_not_a_tip_and_rounded_boost_can_be_a_noop():
    report = _assess(_graph(form=(0, 0), phase=(Q(1, 2**200), 0)))
    assert not report.single_member_source_orbit_equal
    assert report.single_member_source_orbit_status == "obstructed_at_source"
    assert report.first_member.form_phase_correlation_bounds.lo <= 0
    assert report.first_member.form_phase_correlation_bounds.hi > 0

    noops = _assess(boost=Q(1, 2**200))
    assert noops.boost > 0
    assert noops.member_proposals == (Q(1, 4), -Q(1, 4))
    assert noops.single_member_source_orbit_equal
    assert noops.single_member_source_orbit_status == "equal_at_source_both_noops"
    assert noops.whole_pair.effective_form_increments == (0, 0)


def test_increment_adapter_rebuilds_proposals_symmetry_and_target_indices():
    report = _assess()
    poisoned = replace(
        report,
        source_state=replace(
            report.source_state,
            support_symmetry=replace(
                report.source_state.support_symmetry,
                pair_indices=((1, 0),),
                pair_swap_symmetry=(False,),
            ),
        ),
        first_member=replace(
            report.first_member,
            effective_form_increments=(Q(0), Q(0)),
            form_pair=(Q(0), Q(0)),
        ),
        member_proposals=(Q(0), Q(0)),
    )
    rebuilt = poisoned.form_increment(outcome="first_member")
    assert rebuilt.increments == (Q(1, 8), Q(0))
    assert rebuilt.weighted_form_change == Q(1, 8)
    assert rebuilt.closed_flow_endpoint_obstructed
    assert rebuilt == report.form_increment(outcome="first_member")


def test_increment_adapter_rebuilds_changed_primitives_and_effective_clip():
    report = _assess()
    changed = replace(report, boost=Q(1, 2)).form_increment(outcome="whole_pair")
    assert changed.increments == (Q(1, 2), Q(1, 2))
    assert changed.weighted_form_change == 1

    comparison = replace(report.comparison, epi=(Q(7, 8), Q(1)))
    source = replace(
        report.source_state,
        support_symmetry=replace(
            report.source_state.support_symmetry, comparison=comparison
        ),
    )
    clipped = replace(report, source_state=source, boost=Q(1, 2)).form_increment(
        outcome="whole_pair"
    )
    assert clipped.increments == (Q(1, 8), Q(0))
    assert clipped.weighted_form_change == Q(1, 8)

    policy = report.clip_policy
    smaller = replace(report, clip_policy=(policy[0], Q(1, 4), *policy[2:]))
    assert smaller.form_increment(outcome="whole_pair").increments == (0, Q(1, 8))


def test_increment_adapter_normalizes_scalar_premises_before_using_them():
    report = _assess()
    source = replace(
        report.source_state,
        support_symmetry=replace(
            report.source_state.support_symmetry,
            comparison=replace(
                report.comparison,
                epi=(0.25, -0.25),
                phase=(0.25, -0.25),
                capacity=(1.0, 1.0),
            ),
        ),
    )
    changed = replace(report, source_state=source, boost=0.125)
    assert changed.form_increment(outcome="whole_pair") == report.form_increment(
        outcome="whole_pair"
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"boost": True},
        {"boost": float("nan")},
        {"pair_index": True},
        {"pair_index": -1},
        {"clip_policy": (False, 1, "hard", 4)},
        {"clip_policy": (-1, float("inf"), "hard", 4)},
        {"clip_policy": (-1, 1, "unrecognized", 4)},
        {"clip_policy": (-1, 1, "hard", True)},
    ],
)
def test_increment_adapter_readmits_event_primitives(changes):
    with pytest.raises(ADMISSION_ERRORS):
        replace(_assess(), **changes).form_increment(outcome="whole_pair")


@pytest.mark.parametrize("capacity", [(True, 1), (1, 2)])
def test_increment_adapter_cannot_inherit_swap_admission_after_capacity_changes(
    capacity,
):
    report = _assess()
    source = replace(
        report.source_state,
        support_symmetry=replace(
            report.source_state.support_symmetry,
            comparison=replace(report.comparison, capacity=capacity),
        ),
    )
    with pytest.raises(ADMISSION_ERRORS):
        replace(report, source_state=source).form_increment(outcome="whole_pair")


def test_increment_adapter_readmits_declared_phase_chart():
    report = _assess()
    for turns in ((True, 0), (0, 1)):
        changed = replace(
            report, source_state=replace(report.source_state, phase_turns=turns)
        )
        with pytest.raises(ADMISSION_ERRORS):
            changed.form_increment(outcome="whole_pair")


def test_whole_member_swap_and_pair_orientation_preserve_the_unmarked_results():
    graph = _graph(form=(Q(3, 4), -Q(1, 4)))
    report = _assess(graph, boost=Q(1, 2))
    swapped = copy.deepcopy(graph)
    for key in ("EPI", "theta"):
        swapped.nodes[0][key], swapped.nodes[1][key] = (
            graph.nodes[1][key],
            graph.nodes[0][key],
        )
    other = _assess(swapped, boost=Q(1, 2))
    reverse = _assess(graph, pairs=((1, 0),), boost=Q(1, 2))
    for changed in (other, reverse):
        assert (
            changed.single_member_source_orbit_equal
            == report.single_member_source_orbit_equal
        )
        assert changed.first_member.form_mean == report.second_member.form_mean
        assert (
            changed.first_member.internal_form_squared
            == report.second_member.internal_form_squared
        )
        assert (
            changed.first_member.form_phase_correlation_bounds
            == report.second_member.form_phase_correlation_bounds
        )
        assert changed.whole_pair.form_mean == report.whole_pair.form_mean
        assert (
            changed.whole_pair.internal_form_squared
            == report.whole_pair.internal_form_squared
        )
        assert (
            changed.whole_pair.form_phase_correlation_bounds
            == report.whole_pair.form_phase_correlation_bounds
        )


def test_one_capture_ignores_unconsumed_lifecycle_and_never_mutates(monkeypatch):
    graph = _graph()
    graph.graph["EPI_LATENT_MAX"] = "unconsumed invalid trigger configuration"
    graph.nodes[0].update(
        _emission_activated=True,
        _structural_lineage={"activation_count": "invalid runtime lineage"},
        epi_history=[{"value": "not a trigger record"}],
    )
    before = _snapshot(graph)
    capture = owner.bound_relational_sine_exchange
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[0])
        return capture(*args, **kwargs)

    monkeypatch.setattr(owner, "bound_relational_sine_exchange", counted)
    report = _assess(graph)
    assert calls == [graph]
    assert _snapshot(graph) == before
    assert not report.runtime_admission_certified
    assert not report.event_occurrence_derived
    assert report.source_state.comparison.epi == (Q(1, 4), -Q(1, 4))
    assert report.source_state.comparison.phase == (Q(1, 4), -Q(1, 4))


@pytest.mark.parametrize(
    "boost", [0, -1, True, float("nan"), float("inf"), Q(1, 2**2000)]
)
def test_invalid_explicit_boost_rejects_without_source_mutation(boost):
    graph = _graph()
    before = _snapshot(graph)
    with pytest.raises(ADMISSION_ERRORS):
        _assess(graph, boost=boost)
    assert _snapshot(graph) == before


@pytest.mark.parametrize("pair_index", [True, -1, 1, Q(0), "0"])
def test_pair_index_is_an_explicit_nonboolean_integer(pair_index):
    with pytest.raises(ADMISSION_ERRORS):
        _assess(pair_index=pair_index)


def test_invalid_second_proposal_rejects_the_complete_detached_comparison():
    graph = _graph(form=(0, -Q(99, 100)))
    graph.graph["CLIP_MODE"] = "soft"
    before = _snapshot(graph)
    with pytest.raises(ADMISSION_ERRORS):
        _assess(graph, boost=Q(1, 10000))
    assert _snapshot(graph) == before


def test_quotient_admission_does_not_ignore_capacity_support_or_phase_chart():
    graph = _graph()
    graph.nodes[1]["nu_f"] = 2
    with pytest.raises(ADMISSION_ERRORS):
        _assess(graph)
    graph = nx.path_graph(4)
    for node in graph:
        graph.nodes[node].update(EPI=0, theta=0, nu_f=1)
    graph.graph["GAMMA"] = {"type": "none"}
    with pytest.raises(ADMISSION_ERRORS):
        _assess(graph, pairs=((0, 1), (2, 3)))

    branch_graph = _graph(phase=(3, -3))
    with pytest.raises(ADMISSION_ERRORS):
        _assess(branch_graph)
    lifted = _assess(branch_graph, phase_turns=(0, 1))
    assert lifted.source_state.phase_turns == (0, 1)
    assert lifted.whole_pair_structural_descent_certified
    assert not lifted.single_member_source_orbit_equal
    with pytest.raises(ADMISSION_ERRORS):
        _assess(reference_model=RelationalExchangeModel(1, phase_domain="regular"))


def test_export_retains_fine_source_and_rejects_opaque_nested_pair_labels(tmp_path):
    report = _assess()
    exported = report.to_dict()
    generic = relational_report_to_dict(report)
    assert generic["schema"] == "tnfr.relational-report.v1"
    assert generic["report_type"] == "SinePairEmissionAssessment"
    assert generic["report"] == exported["report"]
    assert exported["schema"] == "tnfr.relational-sine-pair-emission.v1"
    assert exported["report"]["runtime_admission_certified"] is False
    assert exported["report"]["event_occurrence_derived"] is False
    target = tmp_path / "emission.json"
    export_to_json(report, target)
    assert json.loads(target.read_text(encoding="utf-8")) == exported

    @dataclass(frozen=True)
    class Opaque:
        value: str

    corrupted_symmetry = replace(
        report.source_state.support_symmetry, pairs=((Opaque("node"), 1),)
    )
    corrupted = replace(
        report,
        source_state=replace(report.source_state, support_symmetry=corrupted_symmetry),
    )
    with pytest.raises(ADMISSION_ERRORS):
        corrupted.to_dict()
