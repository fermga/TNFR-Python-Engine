"""Synthetic callback/wiring controls; never evaluate the reserved THOL event."""

from dataclasses import dataclass
from fractions import Fraction as Q
from types import SimpleNamespace

import networkx as nx
import pytest

from tnfr.alias import get_attr, set_attr, set_dnfr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI
from tnfr.mathematics._rational_interval import I as Interval
from tnfr.research import maintained_parent_thol_execution as owner
from tnfr.research.retained_graph_state import restore_recorded_graph_state


@dataclass
class SyntheticMetabolism:
    neighbor_count: int = 2
    mean_neighbor_epi: Q = Q(3, 10)
    epi_gradient: Q = Q(0)
    phase_variance: Interval = Interval(0, 1)
    coupling_strength_mean: Interval = Interval(0, 1)


@dataclass
class SyntheticPrediction:
    child: str = "2_sub_0"
    child_form_bounds: Interval = Interval(Q(1, 8), Q(3, 8))
    child_capacity: Q = Q(121.6)
    child_phase: Q = Q(0)
    raw_pressure_bounds: Interval = Interval(1, 2)
    child_record_operator_step: int = 2
    observed_acceleration: Q = Q(1, 4)
    child_hierarchy_level: int = 1
    child_hierarchy_path: tuple = (2,)
    numerical_allowance: Q = Q(1, 2**44)
    metabolic: object = None

    def __post_init__(self):
        self.metabolic = SyntheticMetabolism()


@dataclass
class SyntheticStage:
    glyph: str = "THOL"
    nodes_processed: int = 1
    schedule: str = "two_phase_jacobi"


def _admitted():
    graph = nx.Graph(_t=3.0, GAMMA={"type": "none"})
    for node, form, phase in ((0, 0.4, 0.1), (1, 0.2, 0.2), (2, 0.3, 0.0)):
        graph.add_node(node, EPI=form, theta=phase, nu_f=128.0 if node == 2 else 1.0)
    graph.add_edges_from(((0, 1), (1, 2), (2, 0)), weight=1.0)
    graph.nodes[2].update(
        epi_time_history=[(1.0, 0.3), (2.0, 0.3), (3.0, 0.3)],
        glyph_history=["VAL"],
        _grammar_u2_debt=1,
    )
    owner.refresh_pressure(graph)
    context = object()
    steps = []

    def step(index):
        steps.append(index)
        return context

    return SimpleNamespace(
        graph=graph,
        sequence=SimpleNamespace(step=step),
        certificate={"synthetic": True},
        context=context,
        steps=steps,
        threshold=Q(0.1),
        receipts=(),
    )


def _fake_reset(
    graph,
    *,
    child_form=0.25,
    change_old_form=False,
    extra_child=False,
    bad_metabolic_count=False,
    bad_lifecycle_step=False,
):
    """Manufacture a callback result solely to exercise consumer wiring."""
    parent = graph.nodes[2]
    graph.add_node(
        "2_sub_0",
        EPI=child_form,
        theta=0.0,
        nu_f=121.6,
        parent_node=2,
        hierarchy_level=1,
        _bifurcation_level=1,
        _hierarchy_path=[2],
        epi_history=[child_form],
        glyph_history=[],
    )
    set_dnfr(graph, "2_sub_0", 0.0)
    parent.update(
        sub_nodes=["2_sub_0"],
        sub_epis=[
            {
                "node_id": "2_sub_0",
                "metabolized": True,
                "timestamp": 2,
                "d2_epi": 0.25,
                "tau": 0.1,
                "bifurcation_level": 1,
                "hierarchy_path": [2],
                "network_signals": {
                    "neighbor_count": 2,
                    "mean_neighbor_epi": 0.3,
                    "epi_gradient": 0.0,
                    "phase_variance": 0.5,
                    "coupling_strength_mean": 0.5,
                },
            }
        ],
        glyph_history=["VAL", "THOL"],
        _grammar_u2_debt=0,
    )
    graph.graph["hierarchy"] = {2: ["2_sub_0"]}
    if bad_metabolic_count:
        parent["sub_epis"][0]["network_signals"]["neighbor_count"] = 3
    if bad_lifecycle_step:
        parent["sub_epis"][0]["timestamp"] = 1
    set_dnfr(graph, 2, 1.25)
    if change_old_form:
        set_attr(graph.nodes[0], ALIAS_EPI, 0.45)
    if extra_child:
        graph.add_node("extra", EPI=0.5, theta=0.0, nu_f=1.0)
        set_dnfr(graph, "extra", 0.0)


@pytest.fixture(autouse=True)
def forbid_real_proposal(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("wiring controls must not propose or execute a real THOL event")

    monkeypatch.setattr(owner.SelfOrganization, "_prepare_execution", forbidden)
    # Genuine retained-source and stale-state admission have their own tests.
    # This suite isolates callback wiring on an intentionally synthetic graph.
    monkeypatch.setattr(owner, "validate_live_parent", lambda admitted: admitted)
    monkeypatch.setattr(
        owner.network_stage, "execute_self_organization_stage", forbidden
    )


def test_dispatches_one_contextual_stage_retains_raw_reset_and_refreshes_successor(
    monkeypatch,
):
    admitted = _admitted()
    calls = []

    def stage(graph, operator, targets, *, sequence_context, compute_delta_nfr):
        calls.append((operator.glyph.value, targets, sequence_context))
        _fake_reset(graph)
        compute_delta_nfr(graph)
        return SyntheticStage()

    monkeypatch.setattr(owner, "execute_network_operator_stage", stage)
    result = owner.evaluate_thol_event(admitted, SyntheticPrediction())
    assert calls == [("THOL", (2,), admitted.context)]
    assert admitted.steps == [1]
    assert result["all_checks_passed"] and all(result["checks"].values())
    raw = restore_recorded_graph_state(result["raw_after"]).graph
    endpoint = restore_recorded_graph_state(result["endpoint"]).graph
    assert get_attr(raw.nodes[2], ALIAS_DNFR) == 1.25
    assert get_attr(endpoint.nodes[2], ALIAS_DNFR) != 1.25
    assert result["edge_storage_jump"] == 0
    assert result["old_component_mean_changes"] == (0, 0)
    assert not result["global_positive_mean_metric_available"]


@pytest.mark.parametrize("handoffs", [0, 2])
def test_missing_or_repeated_handoff_is_an_explicit_failure(monkeypatch, handoffs):
    def stage(graph, operator, targets, *, sequence_context, compute_delta_nfr):
        _fake_reset(graph)
        for _ in range(handoffs):
            compute_delta_nfr(graph)
        return SyntheticStage()

    monkeypatch.setattr(owner, "execute_network_operator_stage", stage)
    with pytest.raises(ValueError, match="handoff|raw pressure"):
        owner.evaluate_thol_event(_admitted(), SyntheticPrediction())


@pytest.mark.parametrize(
    "change,failed_check",
    [
        ({"child_form": 0.5}, "child_amplitude_predicted"),
        ({"change_old_form": True}, "old_primary_coordinates_unchanged"),
        ({"extra_child": True}, "one_predicted_child"),
        ({"bad_metabolic_count": True}, "metabolic_signals_predicted"),
        ({"bad_lifecycle_step": True}, "child_lifecycle_prediction"),
    ],
)
def test_failed_predicted_response_remains_false_without_retry(
    monkeypatch, change, failed_check
):
    count = 0

    def stage(graph, operator, targets, *, sequence_context, compute_delta_nfr):
        nonlocal count
        count += 1
        _fake_reset(graph, **change)
        compute_delta_nfr(graph)
        return SyntheticStage()

    monkeypatch.setattr(owner, "execute_network_operator_stage", stage)
    result = owner.evaluate_thol_event(_admitted(), SyntheticPrediction())
    assert count == 1
    assert result["checks"][failed_check] is False
    assert result["all_checks_passed"] is False
    if change.get("change_old_form"):
        assert result["edge_storage_jump"] is None


def test_unadmitted_gamma_rejects_before_stage_or_pressure_write(monkeypatch):
    admitted = _admitted()
    admitted.graph.graph["GAMMA"] = {"type": "constant", "value": 1.0}

    def forbidden(*args, **kwargs):
        pytest.fail("live forced source must reject before invoking the stage")

    monkeypatch.setattr(owner, "execute_network_operator_stage", forbidden)
    with pytest.raises((ValueError, TypeError, KeyError)):
        owner.evaluate_thol_event(admitted, SyntheticPrediction())
