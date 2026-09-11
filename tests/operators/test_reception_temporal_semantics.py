"""Adversarial checks for Reception's temporal read boundaries."""

from __future__ import annotations

import inspect
import warnings
from copy import deepcopy
from dataclasses import replace

import networkx as nx
import pytest

import tnfr.operators as operators
import tnfr.operators.event_runtime as event_runtime_module
import tnfr.operators.network_stage as network_stage_module
from tnfr.node import NodeNX
from tnfr.errors import TNFRValueError
from tnfr.operators import apply_glyph, apply_glyph_obj
from tnfr.operators._reception_kernel import (
    RECEPTION_PRE_STATE_BOUNDARY,
    RECEPTION_PRESSURE_OBSERVATION_BOUNDARY,
    capture_reception_read_snapshot,
)
from tnfr.operators.definitions import Reception
from tnfr.operators.event_runtime import execute_operator_event_schedule
from tnfr.operators.event_timing import build_operator_event_schedule
from tnfr.operators.metrics_basic import (
    _reception_metrics_from_snapshot,
    reception_metrics,
)
from tnfr.operators.network_analysis.source_detection import (
    detect_emission_sources,
)
from tnfr.operators.preconditions import validate_reception
from tnfr.operators.network_stage import (
    TWO_PHASE_JACOBI,
    NetworkStageResult,
    ReceptionStageObservation,
    _same_reception_value,
    execute_neighbor_stage,
)
from tnfr.types import Glyph, real_scalar_epi


def _graph(values: tuple[float, ...]) -> nx.Graph:
    graph = nx.path_graph(len(values))
    graph.graph.update(
        GLYPH_FACTORS={"EN_mix": 0.5},
        EPI_MIN=-4.0,
        EPI_MAX=4.0,
    )
    for node, epi in enumerate(values):
        graph.nodes[node].update(
            EPI=epi,
            nu_f=1.0,
            delta_nfr=0.2,
            theta=0.0,
            EPI_kind=f"kind-{node}",
            glyph_history=["AL"],
        )
    return graph


def _epi(graph: nx.Graph, node: int) -> float:
    return float(real_scalar_epi(graph.nodes[node]["EPI"]))


def test_direct_metrics_use_one_pre_en_self_loop_snapshot_and_pressure_magnitude(
) -> None:
    graph = _graph((0.0, 1.0))
    graph.add_edge(0, 0)
    graph.nodes[0]["delta_nfr"] = -2.0

    Reception()(
        graph,
        0,
        track_sources=False,
        collect_metrics=True,
        validate_preconditions=False,
    )

    metrics = graph.graph["operator_metrics"][-1]
    assert _epi(graph, 0) == 0.25
    assert metrics["neighbor_count"] == 2
    assert metrics["neighbor_epi_mean"] == 0.5
    assert metrics["reception_read_boundary"] == RECEPTION_PRE_STATE_BOUNDARY
    assert (
        metrics["dnfr_observation_boundary"]
        == RECEPTION_PRESSURE_OBSERVATION_BOUNDARY
    )
    assert (
        metrics["stored_source_metadata_observation_boundary"]
        == RECEPTION_PRESSURE_OBSERVATION_BOUNDARY
    )
    assert metrics["observed_dnfr"] == -2.0
    assert metrics["pressure_magnitude_below_effectiveness_threshold"] is False
    assert metrics["stabilization_effective"] is False
    assert metrics["source_tracking_enabled"] is False
    assert metrics["source_absence_observed"] is None


def test_direct_reception_captures_after_a_mutating_pre_operator_monitor() -> None:
    class MutatingMonitor:
        def before_operator(self, graph, _node) -> None:
            graph.nodes[1]["EPI"] = 0.8

        def after_operator(self, _graph, _node, _operator) -> None:
            return None

    graph = _graph((0.0, 0.2))
    graph.graph["integrity_monitor"] = MutatingMonitor()

    Reception()(
        graph,
        0,
        track_sources=False,
        collect_metrics=True,
        validate_preconditions=False,
    )

    metrics = graph.graph["operator_metrics"][-1]
    assert _epi(graph, 0) == 0.4
    assert metrics["neighbor_epi_mean"] == 0.8
    assert metrics["delta_epi"] == 0.4


def test_stale_prepared_reception_is_rejected_before_nodenx_cache() -> None:
    graph = _graph((0.0, 0.2))
    snapshot = capture_reception_read_snapshot(
        graph,
        0,
        track_sources=False,
    )
    graph.nodes[1]["EPI"] = 0.8
    expected_nodes = deepcopy(dict(graph.nodes(data=True)))
    expected_graph = deepcopy(dict(graph.graph))

    with pytest.raises(RuntimeError, match="prepared Reception state is stale"):
        operators._apply_prepared_reception_glyph(
            graph,
            0,
            Glyph.EN,
            window=7,
            prepared_state=snapshot,
        )

    assert dict(graph.nodes(data=True)) == expected_nodes
    assert dict(graph.graph) == expected_graph
    assert "_node_cache" not in graph.graph


def test_reception_snapshot_seal_rejects_cross_graph_owner_replacement() -> None:
    graph_a = _graph((0.0, 0.2))
    graph_b = deepcopy(graph_a)
    snapshot = capture_reception_read_snapshot(
        graph_a,
        0,
        track_sources=False,
    )
    forged = replace(
        snapshot,
        _read_graph_owner=graph_b,
        _metric_consumer_graph_owner=graph_b,
        _graph_identity=id(graph_b),
        _metric_consumer_graph_identity=id(graph_b),
    )
    expected_nodes = deepcopy(dict(graph_b.nodes(data=True)))
    expected_graph = deepcopy(dict(graph_b.graph))

    assert snapshot._proof_fields_are_intact()
    assert not forged._proof_fields_are_intact()
    with pytest.raises(ValueError, match="belongs to another graph"):
        operators._apply_prepared_reception_glyph(
            graph_b,
            0,
            Glyph.EN,
            window=7,
            prepared_state=forged,
        )
    with pytest.raises(ValueError, match="belongs to another graph"):
        _reception_metrics_from_snapshot(
            graph_b,
            0,
            0.0,
            read_snapshot=forged,
        )

    assert dict(graph_b.nodes(data=True)) == expected_nodes
    assert dict(graph_b.graph) == expected_graph
    assert "_node_cache" not in graph_b.graph


def test_snapshot_factory_rejects_mismatched_metric_consumer_graph() -> None:
    read_graph = _graph((0.0, 0.2))
    consumer_graph = _graph((0.0, 0.8))

    with pytest.raises(ValueError, match="consumer does not match"):
        capture_reception_read_snapshot(
            read_graph,
            0,
            track_sources=False,
            _metric_consumer_graph_owner=consumer_graph,
        )


def test_stage_rejects_tampered_snapshot_seal_without_metrics(
    monkeypatch,
) -> None:
    graph = _graph((0.0, 0.8))
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    before_graph = deepcopy(dict(graph.graph))
    original = network_stage_module._propose_reception

    def tampered_proposal(*args, **kwargs):
        proposal = original(*args, **kwargs)
        object.__setattr__(
            proposal.reception_read_snapshot,
            "_proof_stamp",
            ("tampered",),
        )
        return proposal

    monkeypatch.setattr(
        network_stage_module,
        "_propose_reception",
        tampered_proposal,
    )

    with pytest.raises(RuntimeError, match="proof fields changed"):
        execute_neighbor_stage(
            graph,
            Reception(),
            (0,),
            track_sources=False,
            collect_metrics=False,
        )

    assert dict(graph.nodes(data=True)) == before_nodes
    assert dict(graph.graph) == before_graph


def test_all_graph_backed_en_entry_points_share_self_loop_and_missing_epi(
) -> None:
    template = _graph((0.0, 0.0, 1.0))
    template.add_edge(0, 0)
    template.add_edge(0, 2)
    template.nodes[1].pop("EPI")
    outputs: list[tuple[float, str]] = []

    direct = deepcopy(template)
    Reception()(
        direct,
        0,
        track_sources=False,
        validate_preconditions=False,
    )
    outputs.append((_epi(direct, 0), direct.nodes[0]["EPI_kind"]))

    low_level = deepcopy(template)
    apply_glyph(low_level, 0, Glyph.EN)
    outputs.append((_epi(low_level, 0), low_level.nodes[0]["EPI_kind"]))

    object_level = deepcopy(template)
    apply_glyph_obj(NodeNX.from_graph(object_level, 0), Glyph.EN)
    outputs.append(
        (_epi(object_level, 0), object_level.nodes[0]["EPI_kind"])
    )

    staged = deepcopy(template)
    execute_neighbor_stage(
        staged,
        Reception(),
        (0,),
        track_sources=False,
    )
    outputs.append((_epi(staged, 0), staged.nodes[0]["EPI_kind"]))

    assert outputs == [(1.0 / 6.0, "kind-2")] * 4


def test_directed_reception_reads_incoming_arcs_and_matches_source_causality(
) -> None:
    graph = nx.DiGraph()
    template = _graph((1.0, 0.0))
    graph.graph.update(deepcopy(dict(template.graph)))
    graph.add_nodes_from(
        (node, deepcopy(data))
        for node, data in template.nodes(data=True)
    )
    graph.add_edge(0, 1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = execute_neighbor_stage(
            graph,
            Reception(),
            (0, 1),
            track_sources=True,
        )

    source, receiver = result.reception_observations
    assert _epi(graph, 0) == 1.0
    assert _epi(graph, 1) == 0.5
    assert source.neighbors == ()
    assert source.reception_sources == ()
    assert receiver.neighbors == (0,)
    assert receiver.neighbor_epi_values == (1.0,)
    assert receiver.reception_sources == ((0, 1.0, 1.0),)
    assert len(caught) == 1
    assert "node 0" in str(caught[0].message)


def _asymmetric_reception_graph(
    incoming_epi: float,
    outgoing_epi: float,
) -> nx.DiGraph:
    template = _graph((incoming_epi, 0.0, outgoing_epi))
    graph = nx.DiGraph()
    graph.graph.update(deepcopy(dict(template.graph)))
    graph.add_nodes_from(
        (node, deepcopy(data))
        for node, data in template.nodes(data=True)
    )
    graph.add_edges_from(((0, 1), (1, 2)))
    return graph


def test_directed_preflight_validates_only_incoming_reception_epi() -> None:
    graph = _asymmetric_reception_graph(1.0, float("nan"))

    apply_glyph(graph, 1, Glyph.EN)

    assert _epi(graph, 1) == 0.5

    invalid = _asymmetric_reception_graph(float("nan"), 1.0)
    with pytest.raises(TNFRValueError):
        apply_glyph(invalid, 1, Glyph.EN)

    assert _epi(invalid, 1) == 0.0
    assert "_node_cache" not in invalid.graph


def test_parallel_directed_arcs_do_not_duplicate_reception_input() -> None:
    template = _graph((1.0, 0.0, 0.25))
    graph = nx.MultiDiGraph()
    graph.graph.update(deepcopy(dict(template.graph)))
    graph.add_nodes_from(
        (node, deepcopy(data))
        for node, data in template.nodes(data=True)
    )
    graph.add_edge(0, 1)
    graph.add_edge(0, 1)
    graph.add_edge(1, 2)

    snapshot = capture_reception_read_snapshot(
        graph,
        1,
        track_sources=True,
        max_distance=1,
    )

    assert snapshot.neighbors == (0,)
    assert snapshot.neighbor_epi_values == (1.0,)
    assert snapshot.reception_sources == ((0, 1.0, 1.0),)


def test_nodenx_subclass_uses_directed_reception_snapshot_semantics() -> None:
    class SubNodeNX(NodeNX):
        pass

    graph = _asymmetric_reception_graph(1.0, float("nan"))

    apply_glyph_obj(SubNodeNX.from_graph(graph, 1), Glyph.EN)

    assert _epi(graph, 1) == 0.5


def test_reception_snapshot_supports_legacy_undirected_graphlike() -> None:
    class GraphLike:
        def __init__(self) -> None:
            self.graph = {}
            self.nodes = {
                0: {"EPI": 0.0, "EPI_kind": "target"},
                1: {"EPI": 1.0, "EPI_kind": "source"},
            }

        def neighbors(self, node):
            return (1,) if node == 0 else (0,)

    graph = GraphLike()

    snapshot = capture_reception_read_snapshot(
        graph,
        0,
        track_sources=False,
    )

    assert snapshot.neighbors == (1,)
    assert snapshot.neighbor_epi_mean == 1.0


def test_source_detection_canonicalizes_equal_fresh_graphlike_keys() -> None:
    class Key:
        def __init__(self, token: int) -> None:
            self.token = token

        def __hash__(self) -> int:
            return self.token

        def __eq__(self, other) -> bool:
            return isinstance(other, Key) and self.token == other.token

    class NodeView(dict):
        def __call__(self):
            return iter(self)

    receiver = Key(0)
    source = Key(1)

    class GraphLike:
        def __init__(self) -> None:
            self.nodes = NodeView(
                {
                    receiver: {"EPI": 0.0, "nu_f": 1.0, "theta": 0.0},
                    source: {"EPI": 1.0, "nu_f": 1.0, "theta": 0.0},
                }
            )

        def neighbors(self, node):
            return (Key(1),) if node == receiver else (Key(0),)

    detected = detect_emission_sources(GraphLike(), receiver, max_distance=1)

    assert len(detected) == 1
    assert detected[0][0] is source


@pytest.mark.parametrize("entry_point", ["operator", "glyph", "object", "stage"])
def test_explicit_none_neighbor_epi_is_never_treated_as_missing(
    entry_point: str,
) -> None:
    graph = _graph((0.0, 1.0))
    graph.nodes[1]["EPI"] = None
    node = NodeNX.from_graph(graph, 0) if entry_point == "object" else None
    before = deepcopy(dict(graph.nodes(data=True)))

    with pytest.raises(TNFRValueError, match="neighbor EPI"):
        if entry_point == "operator":
            Reception()(
                graph,
                0,
                track_sources=False,
                validate_preconditions=False,
            )
        elif entry_point == "glyph":
            apply_glyph(graph, 0, Glyph.EN)
        elif entry_point == "object":
            apply_glyph_obj(node, Glyph.EN)
        else:
            execute_neighbor_stage(
                graph,
                Reception(),
                (0,),
                track_sources=False,
            )

    assert dict(graph.nodes(data=True)) == before


def test_stage_preserves_live_neighbor_order_for_tied_kind_selection() -> None:
    template = nx.Graph()
    for node, epi, kind in (
        (2, -1.0, "second"),
        (1, 0.0, "target"),
        (0, 1.0, "first"),
    ):
        template.add_node(
            node,
            EPI=epi,
            nu_f=1.0,
            delta_nfr=0.2,
            theta=0.0,
            EPI_kind=kind,
            glyph_history=["AL"],
        )
    template.add_edges_from(((0, 1), (1, 2)))
    template.graph.update(
        GLYPH_FACTORS={"EN_mix": 0.5},
        EPI_MIN=-4.0,
        EPI_MAX=4.0,
    )
    assert tuple(template.neighbors(1)) == (0, 2)

    direct = deepcopy(template)
    Reception()(
        direct,
        1,
        track_sources=False,
        validate_preconditions=False,
    )
    staged = deepcopy(template)
    result = execute_neighbor_stage(
        staged,
        Reception(),
        (1,),
        track_sources=False,
    )

    assert direct.nodes[1]["EPI_kind"] == "first"
    assert staged.nodes[1]["EPI_kind"] == "first"
    assert result.reception_observations[0].neighbors == (0, 2)


def test_two_phase_metrics_and_seal_retain_stage_start_reads() -> None:
    graph = _graph((0.0, 0.2, 0.51))
    graph.graph["COLLECT_OPERATOR_METRICS"] = True

    result = execute_neighbor_stage(
        graph,
        Reception(),
        (0, 1, 2),
        track_sources=True,
        max_distance=2,
    )

    assert result.schedule == TWO_PHASE_JACOBI
    assert [item["neighbor_epi_mean"] for item in graph.graph["operator_metrics"]] == [
        0.2,
        0.255,
        0.2,
    ]
    observations = result.reception_observations
    assert len(observations) == 3
    assert all(type(item) is ReceptionStageObservation for item in observations)
    assert all(item._proof_fields_are_intact() for item in observations)
    assert [item.neighbor_epi_mean for item in observations] == [0.2, 0.255, 0.2]
    assert observations[0].reception_sources is not None
    assert [item[0] for item in observations[0].reception_sources] == [2]
    assert observations[0].reception_sources_after == (
        observations[0].reception_sources
    )
    assert observations[0].post_state_boundary == (
        "completed_en_stage_before_result"
    )
    assert observations[0].auxiliary_stability_certified is False
    assert observations[2].reception_sources == ()
    assert observations[2].reception_sources_present_after is True
    assert observations[2].reception_sources_after == ()

    forged = replace(observations[0])
    object.__setattr__(forged, "neighbor_epi_mean", 0.3)
    assert not forged._proof_fields_are_intact()
    with pytest.raises(ValueError, match="not intact or ordered"):
        replace(
            result,
            reception_observations=(forged, *observations[1:]),
        )


@pytest.mark.parametrize("legacy", [None, "legacy", ("not-a-triple",)])
def test_disabled_stage_source_tracking_preserves_opaque_legacy_metadata(
    legacy,
) -> None:
    graph = _graph((0.0, 0.2))
    graph.nodes[0]["_reception_sources"] = legacy

    result = execute_neighbor_stage(
        graph,
        Reception(),
        (0,),
        track_sources=False,
    )

    observation = result.reception_observations[0]
    assert graph.nodes[0]["_reception_sources"] == legacy
    assert observation.source_tracking_enabled is False
    assert observation.reception_sources is None
    assert observation.reception_sources_present_after is True
    assert observation.reception_sources_after is None


def test_tracked_sources_deleted_by_pressure_callback_roll_back_stage() -> None:
    graph = _graph((0.0, 0.8))
    before = deepcopy(dict(graph.nodes(data=True)))

    def delete_sources(subject: nx.Graph) -> None:
        subject.nodes[0].pop("_reception_sources", None)

    with pytest.raises(RuntimeError, match="did not commit its source list"):
        execute_neighbor_stage(
            graph,
            Reception(),
            (0,),
            track_sources=True,
            compute_delta_nfr=delete_sources,
        )

    assert dict(graph.nodes(data=True)) == before


def test_invalid_final_source_state_emits_no_absence_warning() -> None:
    graph = _graph((0.0, 0.0))

    def delete_sources(subject: nx.Graph) -> None:
        subject.nodes[0].pop("_reception_sources", None)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="did not commit its source list"):
            execute_neighbor_stage(
                graph,
                Reception(),
                (0,),
                track_sources=True,
                compute_delta_nfr=delete_sources,
            )

    assert caught == []


def _isolated_reception_graph() -> nx.Graph:
    graph = _graph((0.0, 0.1))
    graph.remove_edge(0, 1)
    graph.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
    for node in graph:
        graph.nodes[node]["delta_nfr"] = 0.01
    return graph


def test_direct_reception_emits_one_centralized_source_warning() -> None:
    graph = _isolated_reception_graph()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Reception()(graph, 0, track_sources=True)

    assert len(caught) == 1
    assert "no emission sources detected" in str(caught[0].message)


def test_stage_reception_emits_one_source_warning_after_validation() -> None:
    graph = _isolated_reception_graph()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        execute_neighbor_stage(
            graph,
            Reception(),
            (0,),
            track_sources=True,
            validate_preconditions=True,
        )

    assert len(caught) == 1
    assert "no emission sources detected" in str(caught[0].message)


def test_stage_source_warning_as_error_rolls_back() -> None:
    graph = _isolated_reception_graph()
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    before_graph = deepcopy(dict(graph.graph))

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning, match="no emission sources detected"):
            execute_neighbor_stage(
                graph,
                Reception(),
                (0,),
                track_sources=True,
                validate_preconditions=True,
            )

    assert dict(graph.nodes(data=True)) == before_nodes
    assert dict(graph.graph) == before_graph


@pytest.mark.parametrize(
    "kwargs",
    [
        {"track_sources": 1},
        {"track_sources": "yes"},
        {"max_distance": True},
        {"max_distance": -1},
    ],
)
def test_reception_controls_reject_noncanonical_values_before_writes(kwargs) -> None:
    graph = _graph((0.0, 0.8))
    before = deepcopy(dict(graph.nodes(data=True)))

    with pytest.raises(Exception):
        Reception()(graph, 0, validate_preconditions=False, **kwargs)

    assert dict(graph.nodes(data=True)) == before


def test_public_reception_precondition_admits_isolated_target_without_warning(
) -> None:
    graph = _graph((0.0, 0.2))
    graph.remove_edge(0, 1)
    graph.nodes[0]["delta_nfr"] = 0.01

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_reception(graph, 0)

    assert caught == []


def test_public_glyph_entry_points_do_not_expose_prepared_state_bypass() -> None:
    assert "_prepared_operator_state" not in inspect.signature(apply_glyph).parameters
    assert (
        "_prepared_operator_state"
        not in inspect.signature(apply_glyph_obj).parameters
    )


def test_standalone_metrics_do_not_infer_tracking_from_old_metadata() -> None:
    graph = _graph((0.0, 0.8))
    graph.nodes[0]["_reception_sources"] = [(1, 1.0, 0.8)]

    metrics = reception_metrics(graph, 0, epi_before=0.0)

    assert metrics["source_tracking_enabled"] is None
    assert metrics["sources_observed"] is None
    assert metrics["source_absence_observed"] is None
    assert metrics["stored_source_metadata_present"] is True
    assert metrics["stored_source_metadata_valid"] is True
    assert (
        metrics["stored_source_metadata_observation_boundary"]
        == "metrics_call_live_state"
    )
    assert metrics["dnfr_observation_boundary"] == "metrics_call_live_state"
    assert "read_snapshot" not in inspect.signature(reception_metrics).parameters


@pytest.mark.parametrize("missing_neighbor_epi", [False, True])
def test_standalone_metrics_reuse_canonical_empty_neighbor_policy(
    missing_neighbor_epi: bool,
) -> None:
    graph = _graph((0.7, 0.2))
    if missing_neighbor_epi:
        graph.nodes[1].pop("EPI")
    else:
        graph.remove_edge(0, 1)

    metrics = reception_metrics(graph, 0, epi_before=0.7)

    assert metrics["neighbor_count"] == 0
    assert metrics["neighbor_epi_mean"] == 0.7
    assert metrics["reception_read_boundary"] == "metrics_call_live_state"


def test_standalone_metrics_select_maximum_legacy_source_compatibility() -> None:
    graph = _graph((0.0, 0.8, 0.2))
    graph.nodes[0]["_reception_sources"] = [
        (1, 0.25, 0.8),
        (2, 0.75, 0.2),
    ]

    metrics = reception_metrics(graph, 0, epi_before=0.0)

    assert metrics["stored_source_metadata_valid"] is True
    assert metrics["most_compatible_source"] == 2


@pytest.mark.parametrize("legacy", [None, "legacy", ("not-a-triple",)])
def test_standalone_metrics_treat_opaque_legacy_sources_as_unobserved(
    legacy,
) -> None:
    graph = _graph((0.0, 0.8))
    graph.nodes[0]["_reception_sources"] = legacy

    metrics = reception_metrics(graph, 0, epi_before=0.0)

    assert metrics["source_tracking_enabled"] is None
    assert metrics["num_sources"] == 0
    assert metrics["stored_source_metadata_present"] is True
    assert metrics["stored_source_metadata_valid"] is False
    assert metrics["source_absence_observed"] is None
    assert (
        metrics["stored_source_metadata_observation_boundary"]
        == "metrics_call_live_state"
    )


def test_operator_metrics_label_live_metadata_presence_after_monitor() -> None:
    class RemovingMonitor:
        def before_operator(self, _graph, _node) -> None:
            return None

        def after_operator(self, graph, node, _operator) -> None:
            graph.nodes[node].pop("_reception_sources", None)

    graph = _graph((0.0, 0.8))
    graph.graph["integrity_monitor"] = RemovingMonitor()

    Reception()(
        graph,
        0,
        track_sources=True,
        collect_metrics=True,
        validate_preconditions=False,
    )

    metrics = graph.graph["operator_metrics"][-1]
    assert metrics["source_tracking_enabled"] is True
    assert metrics["stored_source_metadata_present"] is False
    assert (
        metrics["stored_source_metadata_observation_boundary"]
        == RECEPTION_PRESSURE_OBSERVATION_BOUNDARY
    )


def test_snapshot_controls_are_validated_before_graph_reads() -> None:
    unreadable_graph = object()

    with pytest.raises(TypeError, match="source-tracking"):
        capture_reception_read_snapshot(
            unreadable_graph,
            0,
            track_sources=1,
        )
    with pytest.raises(ValueError, match="max_distance"):
        capture_reception_read_snapshot(
            unreadable_graph,
            0,
            track_sources=True,
            max_distance=True,
        )


def test_object_level_invalid_window_precedes_en_neighbor_reads() -> None:
    graph = _graph((0.0, 0.2))
    graph.nodes[1]["EPI"] = float("nan")
    node = NodeNX.from_graph(graph, 0)
    before = deepcopy(dict(graph.nodes(data=True)))

    with pytest.raises(ValueError, match="window"):
        apply_glyph_obj(node, Glyph.EN, window=-1)

    assert dict(graph.nodes(data=True)) == before


def test_reception_evidence_comparison_avoids_hostile_identifier_equality() -> None:
    class HostileNode:
        def __init__(self, token: int) -> None:
            self.token = token

        def __hash__(self) -> int:
            return self.token

        def __eq__(self, _other) -> bool:
            raise AssertionError("node equality must not be called")

    receiver = HostileNode(1)

    assert _same_reception_value((receiver,), (receiver,))


def test_source_tracking_avoids_hostile_identifier_equality() -> None:
    class HostileNode:
        def __init__(self, token: int) -> None:
            self.token = token

        def __hash__(self) -> int:
            return self.token

        def __eq__(self, _other) -> bool:
            raise AssertionError("node equality must not be called")

    receiver = HostileNode(1)
    source = HostileNode(2)
    graph = nx.Graph()
    graph.add_node(
        receiver,
        EPI=0.0,
        EPI_kind="receiver",
        nu_f=1.0,
        theta=0.0,
    )
    graph.add_node(
        source,
        EPI=1.0,
        EPI_kind="source",
        nu_f=1.0,
        theta=0.0,
    )
    graph.add_edge(source, receiver)

    snapshot = capture_reception_read_snapshot(
        graph,
        receiver,
        track_sources=True,
    )

    assert len(snapshot.reception_sources) == 1
    assert snapshot.reception_sources[0][0] is source


def test_executed_glyph_stage_retains_and_seals_reception_observations() -> None:
    graph = _graph((0.0, 1.0))
    graph.graph.update(
        _t=0.0,
        RANDOM_SEED=7,
        _gamma_spec={"type": "none"},
    )
    for node in graph:
        graph.nodes[node]["glyph_history"] = []
    schedule = build_operator_event_schedule(
        ("emission", "reception", "coherence", "silence"),
        start_time=0.0,
        flow_durations=(0.0, 0.0, 0.0, 0.0, 0.0),
    )

    result = execute_operator_event_schedule(
        graph,
        schedule,
        include_stage_certificates=True,
        suppress_birth_warnings=True,
    )

    stages = result.glyph_stage_evidence
    reception_stage = stages[1]
    assert reception_stage.event.glyph is Glyph.EN
    assert len(reception_stage.reception_observations) == 2
    assert reception_stage._proof_fields_are_intact()
    assert all(
        observation._proof_fields_are_intact()
        for observation in reception_stage.reception_observations
    )
    assert all(not stage.reception_observations for stage in stages[2:])

    altered = replace(reception_stage.reception_observations[0])
    object.__setattr__(altered, "target_epi_kind_after", "forged")
    forged_stage = replace(
        reception_stage,
        reception_observations=(
            altered,
            *reception_stage.reception_observations[1:],
        ),
    )
    assert not altered._proof_fields_are_intact()
    assert not forged_stage._proof_fields_are_intact()

    candidate = replace(
        reception_stage.reception_observations[0],
        target_epi_before=(
            reception_stage.reception_observations[0].target_epi_before + 1.0
        ),
        target_epi_after=(
            reception_stage.reception_observations[0].target_epi_after + 1.0
        ),
        _proof_stamp=(),
    )
    read_payload = network_stage_module._reception_observation_read_snapshot(
        candidate,
    )
    object.__setattr__(
        candidate,
        "_read_payload_stamp",
        network_stage_module._reception_read_payload_stamp(read_payload),
    )
    object.__setattr__(
        candidate,
        "_post_state_payload_stamp",
        network_stage_module._reception_observation_post_state_stamp(candidate),
    )
    object.__setattr__(
        candidate,
        "_proof_stamp",
        network_stage_module._reception_stage_observation_stamp(candidate),
    )
    assert candidate._proof_fields_are_intact()
    stage_candidate = replace(
        reception_stage,
        reception_observations=(
            candidate,
            *reception_stage.reception_observations[1:],
        ),
        _proof_stamp=(),
    )
    resealed_stage = replace(
        stage_candidate,
        _proof_stamp=event_runtime_module._executed_glyph_stage_stamp(
            stage_candidate
        ),
    )
    assert not resealed_stage._proof_fields_are_intact()

    original = reception_stage.reception_observations[0]
    forged_sources = ((999, 1.0, 1.0),)
    anchor_attacks = (
        {
            "neighbor_epi_kinds": tuple(
                "forged" for _kind in original.neighbor_epi_kinds
            )
        },
        {
            "reception_sources": forged_sources,
            "reception_sources_after": forged_sources,
        },
    )
    for changes in anchor_attacks:
        child = replace(original, _proof_stamp=(), **changes)
        object.__setattr__(
            child,
            "_proof_stamp",
            network_stage_module._reception_stage_observation_stamp(child),
        )
        assert not child._proof_fields_are_intact()
        outer = replace(
            reception_stage,
            reception_observations=(
                child,
                *reception_stage.reception_observations[1:],
            ),
            _proof_stamp=(),
        )
        object.__setattr__(
            outer,
            "_proof_stamp",
            event_runtime_module._executed_glyph_stage_stamp(outer),
        )
        assert not outer._proof_fields_are_intact()


def test_post_state_anchor_protects_directed_en_without_certificate() -> None:
    graph = nx.DiGraph()
    template = _graph((0.0, 1.0))
    graph.graph.update(deepcopy(dict(template.graph)))
    graph.graph.update(
        _t=0.0,
        RANDOM_SEED=7,
        _gamma_spec={"type": "none"},
    )
    graph.add_nodes_from(
        (node, deepcopy(data))
        for node, data in template.nodes(data=True)
    )
    graph.add_edges_from(((0, 1), (1, 0)))
    for node in graph:
        graph.nodes[node]["glyph_history"] = []
    schedule = build_operator_event_schedule(
        ("emission", "reception", "coherence", "silence"),
        start_time=0.0,
        flow_durations=(0.0, 0.0, 0.0, 0.0, 0.0),
    )

    result = execute_operator_event_schedule(
        graph,
        schedule,
        include_stage_certificates=True,
        suppress_birth_warnings=True,
    )
    stage = result.glyph_stage_evidence[1]
    assert stage.event.glyph is Glyph.EN
    assert stage.certificate is None
    assert stage._proof_fields_are_intact()

    child = replace(
        stage.reception_observations[0],
        target_epi_kind_after="forged",
        _proof_stamp=(),
    )
    object.__setattr__(
        child,
        "_proof_stamp",
        network_stage_module._reception_stage_observation_stamp(child),
    )
    assert not child._proof_fields_are_intact()
    outer = replace(
        stage,
        reception_observations=(
            child,
            *stage.reception_observations[1:],
        ),
        _proof_stamp=(),
    )
    object.__setattr__(
        outer,
        "_proof_stamp",
        event_runtime_module._executed_glyph_stage_stamp(outer),
    )
    assert not outer._proof_fields_are_intact()


def test_legacy_network_stage_result_may_omit_optional_reception_evidence() -> None:
    result = NetworkStageResult(
        operator="reception",
        glyph=Glyph.EN.value,
        schedule=TWO_PHASE_JACOBI,
        nodes_processed=1,
    )

    assert result.reception_observations == ()
