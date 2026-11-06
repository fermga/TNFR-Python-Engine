"""Public Dissonance keeps local OZ and network propagation atomic."""

from __future__ import annotations

import math
from copy import deepcopy

import networkx as nx
import pytest

from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.errors import TNFRValueError
from tnfr.operators.definitions import Coherence, Dissonance, Emission, Silence
from tnfr.operators.grammar_execution import ValidatedSequence
from tnfr.operators.metrics_basic import dissonance_metrics
from tnfr.operators.network_stage import execute_dissonance_stage
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.topology.asymmetry import compute_topological_asymmetry


def _context():
    return ValidatedSequence([Emission(), Dissonance(), Coherence(), Silence()]).step(1)


def _add_state(graph: nx.Graph, node: int, dnfr: float) -> None:
    graph.add_node(
        node,
        **{
            ALIAS_EPI[0]: 0.5,
            ALIAS_VF[0]: 1.0,
            ALIAS_DNFR[0]: dnfr,
            ALIAS_THETA[0]: 0.0,
            "glyph_history": ["AL"],
            "source_glyph": "AL",
        },
    )


def _graph(graph_type: type[nx.Graph] = nx.Graph) -> nx.Graph:
    graph = graph_type(
        GLYPH_FACTORS={"OZ_dnfr_factor": 2.0},
        RANDOM_SEED=7,
    )
    _add_state(graph, 0, 0.2)
    _add_state(graph, 1, 0.1)
    _add_state(graph, 2, 0.1)
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(0, 2, weight=0.5)
    return graph


def _snapshot(graph: nx.Graph) -> tuple[dict, dict, list]:
    return (
        deepcopy(graph.graph),
        {node: deepcopy(dict(data)) for node, data in graph.nodes(data=True)},
        deepcopy(list(graph.edges(data=True))),
    )


def test_static_asymmetric_snapshot_does_not_fabricate_a_topology_change():
    graph = _graph()
    metrics = dissonance_metrics(graph, 0, 0.2, 0.0)
    # Degrees (2, 1, 1) give CV=sqrt(2)/4; all clustering values are zero.
    assert metrics["topological_asymmetry_after"] == pytest.approx(0.15 * math.sqrt(2))
    assert metrics["topological_asymmetry_before"] is None
    assert metrics["topological_asymmetry_delta"] is None
    assert metrics["topological_asymmetry_change_available"] is False
    assert metrics["topological_asymmetry_changed"] is None
    assert metrics["symmetry_disrupted"] is None


def test_topology_score_rejects_absent_center_but_keeps_small_real_neighborhoods():
    graph = nx.Graph([(0, 1)])
    graph.add_node(2)
    assert compute_topological_asymmetry(graph, 0) == 0.0
    assert compute_topological_asymmetry(graph, 2) == 0.0
    with pytest.raises(nx.NetworkXException):
        compute_topological_asymmetry(graph, "absent")


def test_topology_support_failure_is_not_replaced_by_zero(monkeypatch):
    def unavailable_support(*args, **kwargs):
        raise nx.NetworkXError("ego support unavailable")

    monkeypatch.setattr(nx, "ego_graph", unavailable_support)
    with pytest.raises(nx.NetworkXError, match="ego support unavailable"):
        compute_topological_asymmetry(nx.path_graph(3), 1)


@pytest.mark.parametrize("staged", [False, True])
def test_real_oz_event_captures_unchanged_asymmetric_support(staged, monkeypatch):
    graph = _graph()
    before_edges = list(graph.edges(data=True))
    captured = []
    original = Dissonance._capture_state

    def capture(self, current, node):
        captured.append(node)
        return original(self, current, node)

    monkeypatch.setattr(Dissonance, "_capture_state", capture)
    kwargs = {"sequence_context": _context(), "collect_metrics": True}
    if staged:
        execute_dissonance_stage(graph, Dissonance(), (0,), **kwargs)
    else:
        Dissonance()(graph, 0, **kwargs)
    assert captured == [0]
    assert graph.nodes[0][ALIAS_DNFR[0]] == pytest.approx(0.4)
    assert list(graph.edges(data=True)) == before_edges
    metrics = graph.graph["operator_metrics"][-1]
    assert metrics["topological_asymmetry_before"] == pytest.approx(0.15 * math.sqrt(2))
    assert (
        metrics["topological_asymmetry_after"]
        == metrics["topological_asymmetry_before"]
    )
    assert metrics["topological_asymmetry_delta"] == 0.0
    assert metrics["topological_asymmetry_change_available"] is True
    assert metrics["topological_asymmetry_changed"] is False
    assert metrics["symmetry_disrupted"] is False


@pytest.mark.parametrize("staged", [False, True])
def test_oz_nodal_validation_does_not_request_topology_metrics(staged, monkeypatch):
    from tnfr.topology import asymmetry

    graph = _graph(nx.MultiGraph)
    graph.add_edge(0, 1, weight=0.25)
    graph.graph["VALIDATE_NODAL_EQUATION"] = True
    calls = []

    def unexpected_metric(*args, **kwargs):
        calls.append(args)
        pytest.fail("Nodal validation must not request topological metrics")

    monkeypatch.setattr(asymmetry, "compute_topological_asymmetry", unexpected_metric)
    kwargs = {
        "sequence_context": _context(),
        "collect_metrics": False,
        "propagate_to_network": False,
    }
    if staged:
        execute_dissonance_stage(graph, Dissonance(), (0,), **kwargs)
    else:
        Dissonance()(graph, 0, **kwargs)
    assert calls == []
    assert graph.nodes[0][ALIAS_DNFR[0]] == pytest.approx(0.4)
    assert "operator_metrics" not in graph.graph


def test_supplied_preceding_observation_retains_signed_topology_change():
    graph = _graph()
    before = Dissonance()._capture_metrics_state(graph, 0)
    graph.add_edge(1, 2)  # Supplied support event, not a native OZ operation.
    metrics = Dissonance()._collect_metrics(graph, 0, before)
    assert metrics["topological_asymmetry_after"] == 0.0
    assert metrics["topological_asymmetry_delta"] == pytest.approx(-0.15 * math.sqrt(2))
    # The legacy flag is an absolute-change policy, not successful disruption.
    assert metrics["topological_asymmetry_changed"] is True
    assert metrics["symmetry_disrupted"] is True


@pytest.mark.parametrize("value", [True, math.nan, -0.1, 1.1])
def test_invalid_preceding_topology_score_rejects_before_metric_side_effects(value):
    graph = _graph()
    before = _snapshot(graph)
    with pytest.raises(TNFRValueError, match="topological asymmetry"):
        dissonance_metrics(graph, 0, 0.2, 0.0, asymmetry_before=value)
    assert _snapshot(graph) == before


def test_late_invalid_weight_is_rejected_before_local_oz_or_jitter_progress():
    graph = _graph()
    graph.graph.update(OZ_NOISE_MODE=True, OZ_SIGMA=0.2)
    graph[0][2]["weight"] = math.nan
    metrics = []
    graph.graph["operator_metrics"] = metrics
    source_history = graph.nodes[0]["glyph_history"]
    before = _snapshot(graph)

    with pytest.raises(TNFRValueError, match="edge weight.*must be finite"):
        Dissonance()(
            graph,
            0,
            sequence_context=_context(),
            collect_metrics=True,
        )

    assert _snapshot(graph) == before
    assert graph.nodes[0]["glyph_history"] is source_history
    assert graph.graph["operator_metrics"] is metrics
    assert "_rng_jitter_progress" not in graph.nodes[0]
    assert "_node_cache" not in graph.graph


@pytest.mark.parametrize(
    ("key", "value", "kwargs"),
    [
        ("_oz_propagation_events", (), {}),
        ("operator_metrics", (), {"collect_metrics": True}),
    ],
)
def test_invalid_graph_sink_is_rejected_before_local_oz(key, value, kwargs):
    graph = _graph()
    graph.graph[key] = value
    before = _snapshot(graph)

    with pytest.raises(OperatorPreconditionError, match=f"{key} must be a list"):
        Dissonance()(graph, 0, sequence_context=_context(), **kwargs)

    assert _snapshot(graph) == before
    assert "_node_cache" not in graph.graph


@pytest.mark.parametrize("value", [1, 0, "true", None])
@pytest.mark.parametrize("source", ["argument", "graph"])
def test_propagation_switch_requires_an_active_strict_boolean(source, value):
    graph = _graph()
    kwargs = {}
    label = "OZ_ENABLE_PROPAGATION"
    if source == "argument":
        kwargs["propagate_to_network"] = value
        label = "propagate_to_network"
    else:
        graph.graph["OZ_ENABLE_PROPAGATION"] = value
    before = _snapshot(graph)

    with pytest.raises(OperatorPreconditionError, match=f"{label} must be a boolean"):
        Dissonance()(graph, 0, sequence_context=_context(), **kwargs)

    assert _snapshot(graph) == before


def test_explicit_propagation_switch_supersedes_inactive_graph_setting():
    graph = _graph()
    graph.graph["OZ_ENABLE_PROPAGATION"] = "inactive invalid setting"

    Dissonance()(
        graph,
        0,
        sequence_context=_context(),
        propagate_to_network=False,
    )

    assert graph.nodes[0][ALIAS_DNFR[0]] == pytest.approx(0.4)
    assert graph.nodes[1][ALIAS_DNFR[0]] == pytest.approx(0.1)
    assert "_oz_propagation_events" not in graph.graph


def test_success_commits_local_pressure_neighbors_history_metrics_and_telemetry():
    graph = _graph()

    Dissonance()(
        graph,
        0,
        sequence_context=_context(),
        collect_metrics=True,
    )

    assert graph.nodes[0][ALIAS_DNFR[0]] == pytest.approx(0.4)
    assert graph.nodes[1][ALIAS_DNFR[0]] == pytest.approx(0.3)
    assert graph.nodes[2][ALIAS_DNFR[0]] == pytest.approx(0.2)
    assert list(graph.nodes[0]["glyph_history"]) == ["AL", "OZ"]
    assert graph.nodes[0]["source_glyph"] == "OZ"
    assert graph.graph["operator_metrics"][-1]["operator"] == "Dissonance"

    event = graph.graph["_oz_propagation_events"][-1]
    assert event["source"] == 0
    assert event["magnitude"] == pytest.approx(0.2)
    assert set(event["affected_nodes"]) == {1, 2}
    assert event["affected_count"] == 2
    assert graph.nodes[1]["_oz_propagation"][-1]["magnitude"] == pytest.approx(0.2)
    assert graph.nodes[2]["_oz_propagation"][-1]["magnitude"] == pytest.approx(0.1)


def test_noise_plan_consumes_exactly_one_live_jitter_draw():
    graph = _graph()
    graph.graph.update(
        OZ_NOISE_MODE=True,
        OZ_SIGMA=0.2,
        OZ_MIN_PROPAGATION=0.0,
    )
    source_before = graph.nodes[0][ALIAS_DNFR[0]]
    neighbor_before = graph.nodes[1][ALIAS_DNFR[0]]

    Dissonance()(graph, 0, sequence_context=_context())

    magnitude = abs(graph.nodes[0][ALIAS_DNFR[0]] - source_before)
    assert magnitude > 0.0
    assert graph.nodes[0]["_rng_jitter_progress"]["draws"] == 1
    assert graph.nodes[1][ALIAS_DNFR[0]] == pytest.approx(neighbor_before + magnitude)
    assert graph.graph["_oz_propagation_events"][-1]["magnitude"] == pytest.approx(
        magnitude
    )


class _FailingNodeDict(dict):
    fail_next_propagation_write = False

    def __setitem__(self, key, value):
        if key == "_oz_propagation" and self.fail_next_propagation_write:
            self.fail_next_propagation_write = False
            raise RuntimeError("simulated propagation commit failure")
        super().__setitem__(key, value)


class _FailingGraph(nx.Graph):
    node_attr_dict_factory = _FailingNodeDict


class _AppendThenFailList(list):
    fail_next_append = False

    def append(self, value):
        super().append(value)
        if self.fail_next_append:
            self.fail_next_append = False
            raise RuntimeError("simulated graph telemetry failure")


def test_unexpected_neighbor_commit_failure_restores_public_operator_state():
    graph = _graph(_FailingGraph)
    graph.graph.update(
        OZ_NOISE_MODE=True,
        OZ_SIGMA=0.2,
        OZ_MIN_PROPAGATION=0.0,
        RANDOM_SEED=None,
    )
    metrics = [{"prior": True}]
    propagation_events = [{"source": 99, "prior": True}]
    foreign_metadata = {"preserve": []}
    graph.graph["operator_metrics"] = metrics
    graph.graph["_oz_propagation_events"] = propagation_events
    graph.graph["foreign_metadata"] = foreign_metadata
    source_history = graph.nodes[0]["glyph_history"]
    graph.nodes[2].fail_next_propagation_write = True
    before = _snapshot(graph)

    with pytest.raises(RuntimeError, match="simulated propagation commit failure"):
        Dissonance()(
            graph,
            0,
            sequence_context=_context(),
            collect_metrics=True,
        )

    assert _snapshot(graph) == before
    assert graph.nodes[0]["glyph_history"] is source_history
    assert graph.graph["operator_metrics"] is metrics
    assert graph.graph["_oz_propagation_events"] is propagation_events
    assert graph.graph["foreign_metadata"] is foreign_metadata
    assert graph.graph["RANDOM_SEED"] is None
    assert "_rng_jitter_progress" not in graph.nodes[0]
    assert "_node_cache" not in graph.graph


def test_graph_event_append_failure_rolls_back_propagation_and_sink_in_place():
    graph = _graph()
    propagation_events = _AppendThenFailList([{"source": 99}])
    graph.graph["_oz_propagation_events"] = propagation_events
    before = _snapshot(graph)
    propagation_events.fail_next_append = True

    with pytest.raises(RuntimeError, match="simulated graph telemetry failure"):
        Dissonance()(graph, 0, sequence_context=_context())

    assert _snapshot(graph) == before
    assert graph.graph["_oz_propagation_events"] is propagation_events


def test_self_loop_uses_planned_post_local_pressure_as_propagation_base():
    graph = nx.Graph(
        GLYPH_FACTORS={"OZ_dnfr_factor": 2.0},
        RANDOM_SEED=7,
    )
    _add_state(graph, 0, 0.2)
    graph.add_edge(0, 0, weight=1.0)

    Dissonance()(graph, 0, sequence_context=_context())

    assert graph.nodes[0][ALIAS_DNFR[0]] == pytest.approx(0.6)
    assert graph.nodes[0]["_oz_propagation"][-1]["magnitude"] == pytest.approx(0.2)
