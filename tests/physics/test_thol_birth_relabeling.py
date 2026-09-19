"""Exact selected-state relabeling of one causally prepared THOL birth.

The transported parent mark, insertion order and current unlimited-candidate
policy are inputs. This fixture does not establish arbitrary relabeling,
all-state covariance, a spontaneous selector, or future pattern persistence.
Graph caches, object identities, SHA wall-clock latency timestamps and proof
seals are not compared or rewritten. Structural times remain compared exactly.
"""

from collections import deque
from copy import deepcopy
from datetime import datetime
from fractions import Fraction

import networkx as nx
import numpy as np
import pytest

from benchmarks.thol_birth_transport import STEPS, _birth_graph
from tnfr.dynamics.dnfr import default_compute_delta_nfr
from tnfr.dynamics.sampling import update_node_sample
from tnfr.operators import (
    build_operator_event_schedule,
    build_physical_flow_partition,
    execute_operator_event_schedule,
)
from tnfr.operators.definitions import (
    Coherence,
    Coupling,
    Dissonance,
    SelfOrganization,
    Silence,
)
from tnfr.operators.grammar_dynamics import validate_candidate
from tnfr.utils import ensure_node_offset_map


def _literal(value):
    """Compare binary64 values exactly, without treating integers as node IDs."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return ("binary64", value.hex())
    if isinstance(value, Fraction):
        return ("rational", value.numerator, value.denominator)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, deque):
        return ("deque", value.maxlen, tuple(map(_literal, value)))
    if isinstance(value, dict):
        return {key: _literal(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(map(_literal, value))
    raise TypeError(f"unreviewed projected value type: {type(value).__name__}")


def _node_record(data, inverse):
    """Transport declared pointers and explicitly delimit wall-clock telemetry."""
    result = deepcopy(dict(data))
    if "latency_start_time" in result:
        assert datetime.fromisoformat(result["latency_start_time"]).tzinfo is not None
        result["latency_start_time"] = "<wall-clock timestamp outside projection>"
    if "parent_node" in result:
        result["parent_node"] = inverse[result["parent_node"]]
    for field in ("sub_nodes", "_hierarchy_path"):
        if field in result:
            result[field] = [inverse[node] for node in result[field]]
    for record in result.get("sub_epis", ()):
        record["node_id"] = inverse[record["node_id"]]
        record["hierarchy_path"] = [inverse[node] for node in record["hierarchy_path"]]
    return _literal(result)


def _projection(graph, inverse):
    """Audit node attributes except wall time and explicit graph causal fields."""
    assert set(inverse) == set(graph)
    assert len(set(inverse.values())) == len(graph)
    events = deepcopy(graph.graph.get("_oz_propagation_events", []))
    for event in events:
        event["source"] = inverse[event["source"]]
        event["affected_nodes"] = [inverse[n] for n in event["affected_nodes"]]
    metrics = deepcopy(graph.graph.get("operator_metrics", []))
    for metric in metrics:
        for key in ("node", "node_id"):
            if key in metric:
                metric[key] = inverse[metric[key]]
    return {
        "nodes": tuple(inverse[node] for node in graph),
        "node_data": {
            inverse[node]: _node_record(data, inverse)
            for node, data in graph.nodes(data=True)
        },
        "neighbors": {
            inverse[node]: tuple(inverse[n] for n in graph.neighbors(node))
            for node in graph
        },
        "edges": {
            frozenset((inverse[u], inverse[v])): _literal(dict(data))
            for u, v, data in graph.edges(data=True)
        },
        "time": _literal(graph.graph.get("_t")),
        "hierarchy": {
            inverse[parent]: tuple(inverse[child] for child in children)
            for parent, children in graph.graph.get("hierarchy", {}).items()
        },
        "sample": tuple(inverse[n] for n in graph.graph.get("_node_sample", ())),
        "offsets": {
            inverse[node]: offset
            for node, offset in ensure_node_offset_map(graph).items()
        },
        "policy": _literal(
            {
                key: graph.graph.get(key)
                for key in (
                    "SORT_NODES",
                    "RANDOM_SEED",
                    "UM_FUNCTIONAL_LINKS",
                    "UM_BIDIRECTIONAL",
                    "UM_CANDIDATE_COUNT",
                    "UM_CANDIDATE_MODE",
                    "GLYPH_FACTORS",
                    "GLYPH_HYSTERESIS_WINDOW",
                    "EPI_MIN",
                    "EPI_MAX",
                    "CLIP_MODE",
                    "_dnfr_weights",
                )
            }
        ),
        "oz_events": _literal(events),
        "operator_metrics": _literal(metrics),
        # No scheduled operator events occur: THOL is a direct public call.
        "hybrid_event_log": _literal(graph.graph.get("hybrid_event_log", [])),
    }


def _apply(graph, parent, operator):
    admission = validate_candidate(graph, parent, operator.glyph.value)
    assert admission.allowed
    operator(graph, parent, collect_metrics=True)


def _execute_preparation_flow(graph, inverse):
    schedule = build_operator_event_schedule(
        (),
        start_time=0.0,
        flow_durations=(sum(STEPS),),
    )
    partition = build_physical_flow_partition(schedule.intervals[0], STEPS)
    receipt = execute_operator_event_schedule(
        graph,
        schedule,
        method="euler",
        physical_flow_partitions=(partition,),
    )
    # Authenticate each original receipt; never construct a relabeled seal.
    assert receipt.runtime_clock_checked
    assert receipt.whole_schedule_graph_state_atomic
    assert receipt.events == ()
    evidence = receipt.physical_flow_partition_evidence[0]
    assert evidence._proof_fields_are_intact()
    assert not evidence.future_or_repeated_behavior_certified
    return {
        "targets": tuple(inverse[n] for n in receipt.target_nodes),
        "final_time": _literal(receipt.final_time),
        "refresh_calls": receipt.physical_pressure_refresh_callback_invocations,
        "boundaries": tuple(
            (
                _literal(boundary.time),
                _literal(boundary.after.epi),
                _literal(boundary.after.delta_nfr),
                boundary.nonpressure_state_preserved,
            )
            for boundary in evidence.boundary_observations
        ),
        "methods": tuple(
            item.resolved_method for item in evidence.segment_flow_evidence
        ),
        "clipping": tuple(
            item.clipping_applied for item in evidence.segment_flow_evidence
        ),
    }


def _run(mapping):
    # Each run starts from its own current benchmark preparation. Relabeling
    # occurs before histories exist; graph-owned caches validate their owner.
    graph = nx.relabel_nodes(_birth_graph("attached"), mapping, copy=True)
    inverse = {value: key for key, value in mapping.items()}
    parent = mapping[0]
    assert graph.graph["UM_CANDIDATE_COUNT"] == 0
    update_node_sample(graph, step=0)
    checkpoints = {"initial": _projection(graph, inverse)}
    for operator in (Coherence(), Dissonance()):
        _apply(graph, parent, operator)
        checkpoints[operator.name] = _projection(graph, inverse)
    receipt = _execute_preparation_flow(graph, inverse)
    checkpoints["before_birth"] = _projection(graph, inverse)
    assert tuple(graph.nodes[parent]["glyph_history"]) == ("IL", "OZ")
    before_nodes = set(graph)
    _apply(graph, parent, SelfOrganization())
    children = tuple(graph.nodes[parent]["sub_nodes"])
    assert len(children) == 1
    assert set(graph) - before_nodes == set(children)
    # Birth correspondence follows parent and birth ordinal, not child text,
    # geometric resemblance, or a best-fitting node/channel permutation.
    inverse[children[0]] = ("child", 0, 0)
    checkpoints["raw_birth"] = _projection(graph, inverse)
    default_compute_delta_nfr(graph)
    checkpoints["refreshed_birth"] = _projection(graph, inverse)
    update_node_sample(graph, step=1)
    _apply(graph, parent, Coupling())
    checkpoints["raw_coupling"] = _projection(graph, inverse)
    default_compute_delta_nfr(graph)
    checkpoints["refreshed_coupling"] = _projection(graph, inverse)
    assert graph.has_edge(parent, children[0])
    _apply(graph, parent, Silence())
    checkpoints["closure"] = _projection(graph, inverse)
    return checkpoints, receipt, graph, inverse


@pytest.fixture(scope="module")
def relabeled_births():
    return (
        _run({node: node for node in range(8)}),
        _run({node: f"site-{node:02}" for node in range(8)}),
    )


def test_marked_birth_relabels_newborns_hierarchy_and_causal_history(relabeled_births):
    original, relabeled = relabeled_births
    for stage, state in original[0].items():
        for field, expected in state.items():
            if field == "node_data":
                for node, values in expected.items():
                    actual = relabeled[0][stage][field][node]
                    assert values.keys() == actual.keys()
                    for key, value in values.items():
                        assert value == actual[key], (stage, node, key)
            assert expected == relabeled[0][stage][field], (stage, field)
    assert original[1] == relabeled[1]
    birth = original[0]["raw_birth"]
    child = ("child", 0, 0)
    assert len(birth["nodes"]) == 9
    assert birth["node_data"][child]["parent_node"] == 0
    assert birth["node_data"][child]["_hierarchy_path"] == (0,)
    assert birth["hierarchy"][0] == (child,)
    assert child not in birth["sample"]
    assert child in original[0]["raw_coupling"]["sample"]


@pytest.mark.parametrize(
    "change",
    (
        "child_epi",
        "parent_pointer",
        "hierarchy_path",
        "birth_record",
        "edge_weight",
        "physical_history",
        "glyph_mark",
        "sample",
    ),
)
def test_projection_detects_node_edge_pointer_and_history_corruption(
    relabeled_births,
    change,
):
    _, relabeled = relabeled_births
    expected, _, live, inverse = relabeled
    graph = live.copy()
    graph.graph = deepcopy(
        {
            key: value
            for key, value in live.graph.items()
            if key
            in {
                "_t",
                "hierarchy",
                "_node_sample",
                "_oz_propagation_events",
                "operator_metrics",
                "hybrid_event_log",
                *expected["closure"]["policy"],
            }
        }
    )
    for node in graph:
        graph.nodes[node].clear()
        graph.nodes[node].update(deepcopy(live.nodes[node]))
    parent = "site-00"
    child = graph.nodes[parent]["sub_nodes"][0]
    assert _projection(graph, inverse) == expected["closure"]
    if change == "child_epi":
        graph.nodes[child]["EPI"] += 0.125
    elif change == "parent_pointer":
        graph.nodes[child]["parent_node"] = "site-01"
    elif change == "hierarchy_path":
        graph.nodes[child]["_hierarchy_path"] = ["site-01"]
    elif change == "birth_record":
        graph.nodes[parent]["sub_epis"][0]["node_id"] = "site-01"
    elif change == "edge_weight":
        graph.edges[parent, child]["weight"] += 0.125
    elif change == "physical_history":
        history = graph.nodes[parent]["epi_time_history"]
        time, value = history[0]
        history[0] = (time, value + 0.125)
    elif change == "glyph_mark":
        graph.nodes[parent]["glyph_history"][0] = "AL"
    else:
        graph.graph["_node_sample"] = tuple(n for n in graph if n != child)
    assert _projection(graph, inverse) != expected["closure"]


@pytest.mark.parametrize("omit_child", (False, True))
def test_newborn_correspondence_must_be_a_complete_bijection(
    relabeled_births,
    omit_child,
):
    graph, inverse = relabeled_births[1][2:]
    corrupted = dict(inverse)
    child = graph.nodes["site-00"]["sub_nodes"][0]
    if omit_child:
        del corrupted[child]
    else:
        corrupted[child] = 0
    with pytest.raises(AssertionError):
        _projection(graph, corrupted)
