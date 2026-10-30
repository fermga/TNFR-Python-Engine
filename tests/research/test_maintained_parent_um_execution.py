"""Synthetic dispatch/guard controls; never execute the retained UM response."""

from collections import deque
from dataclasses import dataclass
from fractions import Fraction as Q
from types import SimpleNamespace

import networkx as nx
import pytest

from tnfr.alias import get_attr, set_attr, set_dnfr, set_theta, set_vf
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI
from tnfr.mathematics._rational_interval import I
from tnfr.operators import _coupling_stage_kernel
from tnfr.research import maintained_parent_um_execution as owner
from tnfr.research.maintained_parent_um_source import TholSuccessorAdmission
from tnfr.research.maintained_successor_motion import _bound_motion
from tnfr.research.retained_graph_state import (
    record_graph_state,
    restore_recorded_graph_state,
)
from tnfr.utils.cache import DNFR_PREP_STATE_KEY, _graph_cache_manager


@dataclass(frozen=True)
class SyntheticStage:
    operator: str = "coupling"
    glyph: str = "UM"
    schedule: str = "two_phase_jacobi"
    nodes_processed: int = 1


def _source():
    graph = nx.Graph(_t=3.0, GAMMA={"type": "none"})
    for node, form, phase in ((0, 0.4, 0.1), (1, 0.2, 0.2), (2, 0.3, 0.0)):
        graph.add_node(node, EPI=form, theta=phase, nu_f=1.0)
    graph.add_edges_from(((0, 1), (1, 2), (2, 0)), weight=1.0)
    graph.nodes[2]["glyph_history"] = ["VAL", "THOL"]
    owner.refresh_pressure(graph)
    calls, context = [], object()

    def step(index):
        calls.append(index)
        return context

    admitted = TholSuccessorAdmission(
        graph, None, None, None, SimpleNamespace(step=step), ()
    )
    return admitted, calls, context


@pytest.fixture(autouse=True)
def forbid_real_event(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("synthetic wiring tests must not propose or execute a real UM")

    monkeypatch.setattr(owner.Coupling, "_execute", forbidden)
    monkeypatch.setattr(_coupling_stage_kernel, "propose_coupling_stage", forbidden)
    monkeypatch.setattr(owner.network_stage, "execute_coupling_stage", forbidden)
    monkeypatch.setattr(owner, "_source_prediction", lambda graph, supplied: supplied)


def test_dispatches_once_and_returns_primitive_response_before_assessment(monkeypatch):
    admitted, steps, context = _source()
    before = record_graph_state(admitted.graph)
    dispatched, durable_raw = [], []

    def stage(graph, operator, targets, *, sequence_context, compute_delta_nfr):
        dispatched.append((operator.glyph.value, targets, sequence_context))
        graph.nodes[2]["glyph_history"].append("UM")
        set_dnfr(graph, 2, 1.25)
        compute_delta_nfr(graph)
        graph.graph["final_schedule_marker"] = True
        return SyntheticStage()

    monkeypatch.setattr(owner, "execute_network_operator_stage", stage)

    def forbidden_check(*args, **kwargs):
        pytest.fail("the executor must return durable primitives before assessment")

    monkeypatch.setattr(owner, "check_um_endpoint", forbidden_check)
    result = owner.evaluate_um_event(
        admitted,
        frozen_before=before,
        prediction={"synthetic": True},
        on_raw=durable_raw.append,
    )
    assert dispatched == [("UM", (2,), context)]
    assert steps == [2]
    assert set(result) == {
        "schema",
        "before",
        "raw_after",
        "endpoint",
        "prediction",
        "stage",
        "retained_source_receipts",
    }
    assert durable_raw == [result["raw_after"]]
    raw = restore_recorded_graph_state(result["raw_after"]).graph
    endpoint = restore_recorded_graph_state(result["endpoint"]).graph
    assert get_attr(raw.nodes[2], ALIAS_DNFR) == 1.25
    assert get_attr(endpoint.nodes[2], ALIAS_DNFR) != 1.25
    assert "final_schedule_marker" not in raw.graph
    assert endpoint.graph["final_schedule_marker"] is True


def test_stale_live_source_rejected_before_dispatch(monkeypatch):
    admitted, _, _ = _source()
    before = record_graph_state(admitted.graph)
    set_attr(admitted.graph.nodes[0], ALIAS_EPI, 0.5)
    monkeypatch.setattr(
        owner,
        "execute_network_operator_stage",
        lambda *args, **kwargs: pytest.fail("stale source reached dispatcher"),
    )
    with pytest.raises(ValueError, match="prospectively frozen"):
        owner.evaluate_um_event(admitted, frozen_before=before, prediction={})


@pytest.mark.parametrize("handoffs", [0, 2])
def test_missing_or_repeated_handoff_rejected(monkeypatch, handoffs):
    admitted, _, _ = _source()
    records = []

    def stage(graph, operator, targets, *, sequence_context, compute_delta_nfr):
        for _ in range(handoffs):
            compute_delta_nfr(graph)
        return SyntheticStage()

    monkeypatch.setattr(owner, "execute_network_operator_stage", stage)
    with pytest.raises(ValueError, match="handoff"):
        owner.evaluate_um_event(
            admitted,
            frozen_before=record_graph_state(admitted.graph),
            prediction={},
            on_raw=records.append,
        )
    assert len(records) == min(handoffs, 1)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda graph: set_attr(graph.nodes[0], ALIAS_EPI, 0.45),
        lambda graph: graph.nodes[0].update(theta=0.125),
        lambda graph: graph.nodes[0].update(nu_f=2.0),
        lambda graph: graph.nodes[2]["glyph_history"].append("IL"),
        lambda graph: graph.nodes[2].update(epi_time_history=[(3.0, 0.3)]),
        lambda graph: graph.graph.update(_t=4.0),
        lambda graph: graph.graph.update(_trig_version=100),
        lambda graph: graph.edges[0, 1].update(weight=2.0),
    ],
)
def test_raw_record_survives_forbidden_refresh_mutation(monkeypatch, mutation):
    admitted, _, _ = _source()
    records = []

    def stage(graph, operator, targets, *, sequence_context, compute_delta_nfr):
        set_dnfr(graph, 2, 1.25)
        compute_delta_nfr(graph)
        return SyntheticStage()

    monkeypatch.setattr(owner, "execute_network_operator_stage", stage)
    monkeypatch.setattr(owner, "refresh_pressure", mutation)
    with pytest.raises(ValueError, match="non-pressure"):
        owner.evaluate_um_event(
            admitted,
            frozen_before=record_graph_state(admitted.graph),
            prediction={},
            on_raw=records.append,
        )
    assert len(records) == 1
    raw = restore_recorded_graph_state(records[0]).graph
    assert get_attr(raw.nodes[2], ALIAS_DNFR) == 1.25
    assert raw.graph["_t"] == 3.0
    assert raw.nodes[2]["glyph_history"] == ["VAL", "THOL"]


def test_farthest_defect_is_stronger_than_widened_box_membership():
    ideal, allowance, observed = I(Q(1, 4), Q(3, 8)), Q(1, 8), Q(7, 16)
    assert ideal.lo - allowance <= observed <= ideal.hi + allowance
    assert owner._farthest_defect(observed, ideal) == Q(3, 16) > allowance
    assert owner._farthest_defect(Q(5, 16), ideal) == Q(1, 16)


@pytest.mark.parametrize("count", [True, 0, 2, 1.0])
def test_stage_count_requires_exact_single_target_integer(count):
    stage = dict(
        operator="coupling",
        glyph="UM",
        schedule="two_phase_jacobi",
        nodes_processed=count,
    )
    assert not owner._stage_observation(stage)


def test_stage_dict_and_dataclass_are_both_admitted():
    stage = SyntheticStage()
    assert owner._stage_observation(stage)
    assert owner._stage_observation(owner.encode_result(stage))


def _synthetic_checker_inputs(*, forms=None, old_weight=1.0):
    """An unrelated zero-winding cycle, with a deliberately supplied reset.

    Source admission is mocked by this module's fixture. The expected scalar
    changes below are dyadic inputs, not values taken from a retained response
    or from the UM implementation. The real checker still rebuilds both rows,
    every edge work term and every context-preservation predicate.
    """
    before = nx.Graph(_t=float(owner.TIMES[-1]), GAMMA={"type": "none"})
    nodes = tuple(range(20)) + (owner.CHILD,)
    forms = (0.25,) * len(nodes) if forms is None else forms
    for node, form in zip(nodes, forms, strict=True):
        before.add_node(node, EPI=form, theta=0.0, nu_f=1.0, glyph_history=[])
        set_dnfr(before, node, 0.0)
    before.add_edges_from(((i, (i + 1) % 20) for i in range(20)), weight=old_weight)
    before.nodes[2].update(
        nu_f=2.0,
        glyph_history=deque(("VAL", "THOL"), maxlen=7),
        _operator_step=2,
        _grammar_u2_debt=0,
        _grammar_prior_coherence=False,
        source_glyph="THOL",
        epi_time_history=((float(owner.TIMES[-1]), forms[2]),),
        sub_nodes=[owner.CHILD],
    )
    before.nodes[owner.CHILD].update(
        parent_node=2, hierarchy_level=1, epi_history=[forms[-1]]
    )
    before.nodes[0]["retained_marker"] = 1
    before.graph.update(hierarchy={2: [owner.CHILD]}, _vfmax=2.0, _vfmax_node=2)
    old_edges = tuple(sorted(tuple(sorted(edge)) for edge in before.edges))
    new_edges = ((2, 20),)
    edges = tuple(sorted(old_edges + new_edges))
    epsilon = owner.NUMERICAL_ALLOWANCE
    phases = tuple(I(Q(1, 8) if index == 2 else 0) for index in range(21))
    capacities = tuple(I(Q(3, 2) if index == 2 else 1) for index in range(21))
    weights = tuple(
        I(Q(1, 2) if edge in new_edges else Q(old_weight)) for edge in edges
    )
    conditional_phases = tuple(
        value + I(-epsilon, epsilon) if index == 2 else value
        for index, value in enumerate(phases)
    )
    conditional_capacities = tuple(
        value + I(-epsilon, epsilon) if index == 2 else value
        for index, value in enumerate(capacities)
    )
    conditional_weights = tuple(
        value + I(-epsilon, epsilon) if edge in new_edges else value
        for edge, value in zip(edges, weights, strict=True)
    )
    reset = SimpleNamespace(
        nodes=nodes,
        epi=tuple(map(Q, forms)),
        phase_before=(Q(0),) * 21,
        phase_after=phases,
        capacity_before=tuple(Q(2 if i == 2 else 1) for i in range(21)),
        capacity_after=capacities,
        raw_pressure_before=(Q(0),) * 21,
        raw_pressure_after=tuple(I(Q(1, 8) if i == 2 else 0) for i in range(21)),
        old_edges=old_edges,
        old_conductances=(Q(old_weight),) * 20,
        new_edges=new_edges,
        new_conductances=(I(Q(1, 2)),),
        phase_changed_indices=(2,),
        capacity_changed_indices=(2,),
    )
    chamber = SimpleNamespace(
        reset=reset,
        edges=edges,
        ideal=owner._certify_chamber(phases, edges),
        conditional=owner._certify_chamber(conditional_phases, edges),
        conditional_capacities=conditional_capacities,
        conditional_conductances=conditional_weights,
    )
    conditional = _bound_motion(
        epi=reset.epi,
        phases=conditional_phases,
        capacities=conditional_capacities,
        edges=edges,
        conductances=conditional_weights,
        shifts=chamber.ideal.shifts,
        horizon=Q(1, 2),
    )
    prediction = SimpleNamespace(chamber=chamber, conditional=conditional)
    raw = restore_recorded_graph_state(record_graph_state(before)).graph
    set_theta(raw, 2, 0.125)
    set_vf(raw, 2, 1.5)
    set_dnfr(raw, 2, 0.125)
    raw.add_edge(2, owner.CHILD, weight=0.5)
    raw.graph.update(_edge_version=1, _dnfr_prep_dirty=True)
    raw.nodes[2]["glyph_history"].append("UM")
    raw.nodes[2].update(_operator_step=3, source_glyph="UM")
    raw._last_operator_applied = "coupling"
    endpoint = restore_recorded_graph_state(record_graph_state(raw)).graph
    owner.refresh_pressure(endpoint)
    endpoint.graph[owner.network_stage.STAGE_SCHEDULE_KEY] = owner.encode_result(
        SyntheticStage()
    )
    endpoint.graph[owner.network_stage.STAGE_CONTRACT_KEY] = (
        owner.stage_schedule_metadata("coupling", observed_schedule="two_phase_jacobi")
    )
    return before, raw, endpoint, prediction


def test_complete_checker_rebuilds_independent_synthetic_reset_rows_and_work():
    before, raw, endpoint, prediction = _synthetic_checker_inputs()
    result = owner.check_um_endpoint(
        before, raw, endpoint, prediction=prediction, stage=SyntheticStage()
    )
    assert result["all_checks_passed"], result["checks"]
    assert all(
        error == 0 for _, error in result["changed_output_error_bounds"]["phase"]
    )
    assert all(value == {"lo": 0, "hi": 0} for value in result["old_edge_form_work"])
    assert result["actual_work"]["new_form_total"] == {"lo": 0, "hi": 0}
    assert result["actual_total_work"]["lo"] > 0
    assert not result["complete_word_executed"]


def test_old_form_work_cancels_exactly_before_interval_enclosure():
    forms = (0.1, 0.3) * 10 + (0.25,)
    before, raw, endpoint, prediction = _synthetic_checker_inputs(
        forms=forms, old_weight=0.1
    )
    # A nonzero old energy exceeds the 128-bit enclosure grid. Enclosing its
    # copies separately cannot prove equality, despite unchanged primitives.
    energy = Q(0.1) * (Q(0.1) - Q(0.3)) ** 2 / 2
    assert energy > 0 and I(energy).width > 0
    result = owner.check_um_endpoint(
        before, raw, endpoint, prediction=prediction, stage=SyntheticStage()
    )
    assert result["all_checks_passed"], result["checks"]
    assert all(value == {"lo": 0, "hi": 0} for value in result["old_edge_form_work"])


def test_old_form_work_retains_a_real_nonzero_conductance_change():
    before, raw, endpoint, prediction = _synthetic_checker_inputs(
        forms=(0.1, 0.3) * 10 + (0.25,), old_weight=0.1
    )
    raw.edges[0, 1]["weight"] = endpoint.edges[0, 1]["weight"] = 0.125
    expected = (Q(0.125) - Q(0.1)) * (Q(0.1) - Q(0.3)) ** 2 / 2
    result = owner.check_um_endpoint(
        before, raw, endpoint, prediction=prediction, stage=SyntheticStage()
    )
    observed = result["old_edge_form_work"][0]
    assert 0 < observed["lo"] <= expected <= observed["hi"]
    assert not result["checks"]["old_edge_form_work_zero"]
    assert not result["checks"]["old_edge_attributes_unchanged"]
    assert not result["all_checks_passed"]


@pytest.mark.parametrize("flag", [None, False, 1, "true"])
def test_new_support_requires_exact_preparation_invalidation_flag(flag):
    before, raw, endpoint, prediction = _synthetic_checker_inputs()
    for graph in (raw, endpoint):
        if flag is None:
            graph.graph.pop("_dnfr_prep_dirty")
        else:
            graph.graph["_dnfr_prep_dirty"] = flag
    result = owner.check_um_endpoint(
        before, raw, endpoint, prediction=prediction, stage=SyntheticStage()
    )
    assert result["checks"]["pressure_only_refresh_boundary"]
    assert not result["checks"]["declared_graph_context_preserved"]
    assert not result["all_checks_passed"]


def test_preparation_invalidation_flag_remains_protected_during_refresh():
    before, raw, endpoint, prediction = _synthetic_checker_inputs()
    endpoint.graph["_dnfr_prep_dirty"] = False
    result = owner.check_um_endpoint(
        before, raw, endpoint, prediction=prediction, stage=SyntheticStage()
    )
    assert not result["checks"]["pressure_only_refresh_boundary"]
    assert not result["checks"]["declared_graph_context_preserved"]


def test_topology_invalidation_does_not_admit_other_graph_context_changes():
    before, raw, endpoint, prediction = _synthetic_checker_inputs()
    raw.graph["invented_context"] = endpoint.graph["invented_context"] = True
    result = owner.check_um_endpoint(
        before, raw, endpoint, prediction=prediction, stage=SyntheticStage()
    )
    assert result["checks"]["pressure_only_refresh_boundary"]
    assert not result["checks"]["declared_graph_context_preserved"]
    assert not result["all_checks_passed"]


def test_checker_rejects_invented_context_even_with_valid_physical_endpoint():
    before, raw, endpoint, prediction = _synthetic_checker_inputs()
    endpoint.nodes[owner.CHILD]["epi_time_history"] = [(float(owner.TIMES[-1]), 0.25)]
    result = owner.check_um_endpoint(
        before, raw, endpoint, prediction=prediction, stage=SyntheticStage()
    )
    assert not result["all_checks_passed"]
    assert not result["checks"]["pressure_only_refresh_boundary"]
    assert not result["checks"]["child_temporal_baseline_unavailable"]


@pytest.mark.parametrize(
    "node,key", [(owner.CHILD, "hierarchy_level"), (0, "retained_marker")]
)
def test_checker_preserves_metadata_scalar_types_across_the_semantic_jump(node, key):
    before, raw, endpoint, prediction = _synthetic_checker_inputs()
    assert before.nodes[node][key] == 1 and type(before.nodes[node][key]) is int
    # Both post-event records agree, so the pressure-only guard is insufficient.
    raw.nodes[node][key] = endpoint.nodes[node][key] = True
    result = owner.check_um_endpoint(
        before, raw, endpoint, prediction=prediction, stage=SyntheticStage()
    )
    assert result["checks"]["pressure_only_refresh_boundary"]
    assert not result["checks"]["declared_nodal_context_preserved"]
    assert not result["all_checks_passed"]


def test_context_comparison_distinguishes_bool_integer_and_exact_fraction():
    assert not owner._same({"value": True}, {"value": 1})
    assert not owner._same({"value": Q(1)}, {"value": 1})
    assert owner._same({"value": Q(1)}, {"value": Q(1)})


def test_checker_rejects_extra_pressure_alias_under_authoritative_value():
    before, raw, endpoint, prediction = _synthetic_checker_inputs()
    endpoint.nodes[0]["dnfr"] = "invalid shadow"
    result = owner.check_um_endpoint(
        before, raw, endpoint, prediction=prediction, stage=SyntheticStage()
    )
    assert not result["all_checks_passed"]
    assert not result["checks"]["pressure_only_refresh_boundary"]


@pytest.mark.parametrize("alias", ["freq", "last_glyph", "dnfr"])
def test_checker_preserves_unwritten_aliases_during_the_semantic_jump(alias):
    before, raw, endpoint, prediction = _synthetic_checker_inputs()
    raw.nodes[2][alias] = endpoint.nodes[2][alias] = "invalid shadow"
    result = owner.check_um_endpoint(
        before, raw, endpoint, prediction=prediction, stage=SyntheticStage()
    )
    assert result["checks"]["pressure_only_refresh_boundary"]
    assert not result["checks"]["declared_nodal_context_preserved"]
    assert not result["all_checks_passed"]


def test_checker_rejects_contradictory_phase_compatibility_alias():
    before, raw, endpoint, prediction = _synthetic_checker_inputs()
    raw.nodes[2]["phase"] = endpoint.nodes[2]["phase"] = 0.25
    result = owner.check_um_endpoint(
        before, raw, endpoint, prediction=prediction, stage=SyntheticStage()
    )
    assert not result["checks"]["canonical_changed_phase_aliases"]
    assert not result["all_checks_passed"]


def test_live_refresh_guard_preserves_pressure_alias_inventory(monkeypatch):
    admitted, _, _ = _source()

    def stage(graph, operator, targets, *, sequence_context, compute_delta_nfr):
        compute_delta_nfr(graph)
        return SyntheticStage()

    monkeypatch.setattr(owner, "execute_network_operator_stage", stage)
    monkeypatch.setattr(
        owner, "refresh_pressure", lambda graph: graph.nodes[0].update(dnfr=0.0)
    )
    with pytest.raises(ValueError, match="non-pressure"):
        owner.evaluate_um_event(
            admitted, frozen_before=record_graph_state(admitted.graph), prediction={}
        )


def test_live_refresh_with_cache_owner_back_reference_preserves_boundary(monkeypatch):
    admitted, _, _ = _source()
    graph = admitted.graph
    manager = _graph_cache_manager(graph.graph)
    manager.clear(DNFR_PREP_STATE_KEY)
    assert manager._graph_owner is graph.graph

    def stage(graph, operator, targets, *, sequence_context, compute_delta_nfr):
        set_dnfr(graph, 2, 1.25)
        compute_delta_nfr(graph)
        return SyntheticStage()

    monkeypatch.setattr(owner, "execute_network_operator_stage", stage)
    result = owner.evaluate_um_event(
        admitted, frozen_before=record_graph_state(graph), prediction={}
    )
    raw = restore_recorded_graph_state(result["raw_after"]).graph
    endpoint = restore_recorded_graph_state(result["endpoint"]).graph
    assert raw.graph["_dnfrmax"] == 1.25
    assert endpoint.graph["_dnfrmax"] < 1.25
    assert graph.graph["_tnfr_cache_manager"] is manager


def test_live_refresh_still_protects_cache_owned_payload(monkeypatch):
    admitted, _, _ = _source()
    manager = _graph_cache_manager(admitted.graph.graph)
    manager.clear(DNFR_PREP_STATE_KEY)
    original_refresh = owner.refresh_pressure

    def changed_payload(graph):
        original_refresh(graph)
        manager._default_capacity += 1

    def stage(graph, operator, targets, *, sequence_context, compute_delta_nfr):
        compute_delta_nfr(graph)
        return SyntheticStage()

    monkeypatch.setattr(owner, "refresh_pressure", changed_payload)
    monkeypatch.setattr(owner, "execute_network_operator_stage", stage)
    with pytest.raises(ValueError, match="non-pressure"):
        owner.evaluate_um_event(
            admitted, frozen_before=record_graph_state(admitted.graph), prediction={}
        )
