"""Atomicity and domain tests for the public THOL boundary."""

from __future__ import annotations

import math
from copy import deepcopy

import networkx as nx
import pytest

from tnfr.alias import get_attr, get_attr_str
from tnfr.constants.canonical import COUPLING_GENTLE, COUPLING_MODERATE
from tnfr.constants.aliases import (
    ALIAS_D2EPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_EPI_KIND,
    ALIAS_SOURCE_GLYPH,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.node import NodeNX
from tnfr.mathematics import BEPIElement
from tnfr.operators import apply_glyph
from tnfr.operators.definitions import SelfOrganization
from tnfr.operators.metabolism import (
    capture_network_signals,
    metabolize_signals_into_subepi,
)
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.types import ensure_bepi, serialize_bepi


def _graph(*, metabolic: bool = True, propagation: bool = False) -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(
        0,
        **{
            ALIAS_EPI[0]: 0.6,
            ALIAS_VF[0]: 1.0,
            ALIAS_DNFR[0]: 0.2,
            ALIAS_THETA[0]: 0.1,
            ALIAS_D2EPI[0]: 0.2,
            ALIAS_EPI_KIND[0]: "seed-identity",
            ALIAS_SOURCE_GLYPH[0]: "OZ",
            "epi_history": [0.0, 0.1, 0.6],
            "glyph_history": ["OZ"],
        },
    )
    graph.add_node(
        1,
        **{
            ALIAS_EPI[0]: 0.4,
            ALIAS_VF[0]: 1.0,
            ALIAS_DNFR[0]: 0.1,
            ALIAS_THETA[0]: 0.12,
            "epi_history": [0.3, 0.35, 0.4],
        },
    )
    graph.add_edge(0, 1)
    graph.graph["THOL_METABOLIC_ENABLED"] = metabolic
    graph.graph["THOL_PROPAGATION_ENABLED"] = propagation
    return graph


def _plain_state(graph: nx.Graph) -> tuple[object, ...]:
    graph_data = {
        key: deepcopy(value)
        for key, value in graph.graph.items()
        if key not in {"_node_cache", "_node_cache_weak", "integrity_monitor"}
    }
    return (
        tuple(graph.nodes),
        tuple(graph.edges),
        {node: deepcopy(dict(data)) for node, data in graph.nodes(data=True)},
        graph_data,
    )


def test_thol_uses_physical_history_and_commits_complete_u5_state() -> None:
    graph = _graph(metabolic=False, propagation=False)
    graph.nodes[0]["epi_history"] = [0.6, 0.6, 0.6]
    graph.nodes[0]["epi_time_history"] = [
        (0.0, 0.0),
        (1.0, 0.1),
        (2.0, 0.6),
    ]
    graph.nodes[0][ALIAS_THETA[0]] = math.tau + 0.25
    graph.graph.update(
        THOL_METABOLIC_GRADIENT_WEIGHT="inactive",
        THOL_METABOLIC_COMPLEXITY_WEIGHT=float("nan"),
        THOL_MIN_COUPLING_FOR_PROPAGATION="inactive",
        THOL_PROPAGATION_ATTENUATION=-1.0,
    )

    SelfOrganization()(graph, 0, tau=0.1, collect_metrics=True)

    record = graph.nodes[0]["sub_epis"][0]
    child = record["node_id"]
    assert get_attr(graph.nodes[0], ALIAS_EPI) == pytest.approx(0.6)
    assert record["vf"] == pytest.approx(get_attr(graph.nodes[child], ALIAS_VF))
    assert record["d2_epi"] == pytest.approx(0.4)
    assert graph.nodes[child]["parent_node"] == 0
    assert graph.nodes[child]["_hierarchy_path"] == [0]
    assert 0.0 <= get_attr(graph.nodes[child], ALIAS_THETA) < math.tau
    assert get_attr_str(graph.nodes[0], ALIAS_EPI_KIND) == "seed-identity"
    assert get_attr_str(graph.nodes[0], ALIAS_SOURCE_GLYPH) == "THOL"
    assert graph.graph["hierarchy"][0] == [child]
    assert graph.graph["operator_metrics"][-1]["nested_epi_count"] == 1


@pytest.mark.parametrize("epi", [ensure_bepi(0.6), serialize_bepi(0.6)])
def test_public_thol_accepts_canonical_bepi_form_state(epi) -> None:
    graph = _graph(metabolic=False, propagation=False)
    graph.nodes[0][ALIAS_EPI[0]] = epi

    SelfOrganization()(graph, 0, tau=0.1)

    assert len(graph.nodes[0]["sub_epis"]) == 1
    assert graph.nodes[0]["sub_epis"][0]["epi"] == pytest.approx(0.18)


def test_public_thol_rejects_nonuniform_bepi_without_flattening_form() -> None:
    graph = _graph(metabolic=False, propagation=False)
    rich = BEPIElement((0.5, 0.6), (0.5, 0.5), (0.0, 1.0))
    graph.nodes[0][ALIAS_EPI[0]] = serialize_bepi(rich)
    before = _plain_state(graph)

    with pytest.raises(OperatorPreconditionError, match="signed scalar embedding"):
        SelfOrganization()(graph, 0, tau=0.1)

    assert _plain_state(graph) == before
    assert "_node_cache" not in graph.graph


def test_public_thol_accepts_state_after_an_epi_writing_operator() -> None:
    graph = _graph(metabolic=False, propagation=False)
    graph.nodes[0][ALIAS_EPI[0]] = ensure_bepi(0.5)
    apply_glyph(graph, 0, "AL")
    graph.nodes[0]["epi_history"] = [0.0, 0.1, get_attr(graph.nodes[0], ALIAS_EPI)]
    graph.nodes[0]["glyph_history"].append("OZ")

    SelfOrganization()(graph, 0, tau=0.1)

    assert len(graph.nodes[0]["sub_epis"]) == 1
    assert get_attr_str(graph.nodes[0], ALIAS_EPI_KIND) == "seed-identity"


@pytest.mark.parametrize(
    "case",
    [
        "epi",
        "vf",
        "dnfr",
        "theta",
        "physical_history",
        "tau",
        "max_depth",
        "metabolic_flag",
        "gradient_weight",
        "complexity_weight",
        "propagation_flag",
        "epi_max",
        "hierarchy",
        "sub_epi",
    ],
)
def test_active_invalid_state_config_or_proposal_is_rejected_prewrite(
    case: str,
) -> None:
    graph = _graph()
    kwargs: dict[str, object] = {"tau": 0.1}
    if case == "epi":
        graph.nodes[0][ALIAS_EPI[0]] = float("nan")
    elif case == "vf":
        graph.nodes[0][ALIAS_VF[0]] = -1.0
    elif case == "dnfr":
        graph.nodes[0][ALIAS_DNFR[0]] = float("inf")
    elif case == "theta":
        graph.nodes[0][ALIAS_THETA[0]] = float("nan")
    elif case == "physical_history":
        graph.nodes[0]["epi_time_history"] = [
            (0.0, 0.0),
            (0.0, 0.1),
            (2.0, 0.6),
        ]
    elif case == "tau":
        kwargs["tau"] = True
    elif case == "max_depth":
        graph.graph["THOL_MAX_BIFURCATION_DEPTH"] = 1.5
    elif case == "metabolic_flag":
        graph.graph["THOL_METABOLIC_ENABLED"] = 1
    elif case == "gradient_weight":
        graph.graph["THOL_METABOLIC_GRADIENT_WEIGHT"] = float("nan")
    elif case == "complexity_weight":
        graph.graph["THOL_METABOLIC_COMPLEXITY_WEIGHT"] = 1.1
    elif case == "propagation_flag":
        graph.graph["THOL_PROPAGATION_ENABLED"] = "yes"
    elif case == "epi_max":
        graph.graph["EPI_MIN"] = 0.0
        graph.graph["EPI_MAX"] = -1.0
    elif case == "hierarchy":
        graph.graph["hierarchy"] = []
    elif case == "sub_epi":
        graph.nodes[0]["sub_epis"] = [{"epi": float("nan")}]

    before = _plain_state(graph)
    with pytest.raises((ValueError, OperatorPreconditionError)):
        SelfOrganization()(graph, 0, **kwargs)
    assert _plain_state(graph) == before
    assert "_node_cache" not in graph.graph


def test_disabled_bifurcation_channels_do_not_read_their_factors() -> None:
    graph = _graph(metabolic=False, propagation=False)
    graph.graph.update(
        THOL_METABOLIC_GRADIENT_WEIGHT=object(),
        THOL_METABOLIC_COMPLEXITY_WEIGHT=float("nan"),
        THOL_MIN_COUPLING_FOR_PROPAGATION=object(),
        THOL_PROPAGATION_ATTENUATION=float("inf"),
    )

    SelfOrganization()(graph, 0, tau=0.1)

    assert len(graph.nodes[0]["sub_epis"]) == 1
    assert graph.nodes[0]["sub_epis"][0]["metabolized"] is False
    assert "thol_propagations" not in graph.graph


def test_empty_thol_window_does_not_read_bifurcation_configuration() -> None:
    graph = _graph()
    graph.nodes[0]["epi_history"] = [0.0, 0.1, 0.2]
    graph.graph.update(
        THOL_METABOLIC_ENABLED="inactive",
        THOL_PROPAGATION_ENABLED="inactive",
        THOL_MAX_BIFURCATION_DEPTH="inactive",
        THOL_METABOLIC_GRADIENT_WEIGHT="inactive",
    )

    SelfOrganization()(graph, 0, tau=0.1)

    assert "sub_epis" not in graph.nodes[0]
    assert list(graph.nodes[0]["glyph_history"])[-1] == "THOL"


def test_precondition_telemetry_is_deferred_until_success() -> None:
    graph = _graph(metabolic=False, propagation=False)
    graph.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
    graph.graph["THOL_MAX_BIFURCATION_DEPTH"] = "late-invalid"
    before = _plain_state(graph)

    with pytest.raises(OperatorPreconditionError):
        SelfOrganization()(graph, 0, tau=0.1)

    assert _plain_state(graph) == before
    assert "_mutation_context" not in graph.nodes[0]
    assert "_thol_no_bifurcation_expected" not in graph.nodes[0]


@pytest.mark.parametrize("invalid_boundary", ["metrics", "monitor"])
def test_shared_boundary_is_validated_before_grammar_fallback(
    invalid_boundary: str,
) -> None:
    graph = _graph()
    graph.nodes[0]["glyph_history"] = []
    if invalid_boundary == "metrics":
        graph.graph["operator_metrics"] = ()
    else:
        graph.graph["integrity_monitor"] = object()
    before = _plain_state(graph)

    with pytest.raises(OperatorPreconditionError):
        SelfOrganization()(graph, 0, collect_metrics=True)

    assert _plain_state(graph) == before
    assert "_node_cache" not in graph.graph


def test_successful_precondition_telemetry_uses_the_same_acceleration() -> None:
    graph = _graph(metabolic=True, propagation=False)
    graph.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = True
    graph.nodes[0]["epi_history"] = [0.0, 0.1, 0.2]

    SelfOrganization()(graph, 0, tau=0.1)

    assert graph.nodes[0]["_thol_no_bifurcation_expected"] is True
    assert graph.nodes[0]["_mutation_context"]["destabilizer_operator"] == (
        "dissonance"
    )


def test_late_commit_error_restores_topology_caches_and_metadata(monkeypatch) -> None:
    graph = _graph()
    graph.graph["operator_metrics"] = [{"preserved": True}]
    cached_node = NodeNX(graph, 0)
    cache = graph.graph["_node_cache"]
    cache_items = tuple(cache.items())
    before = _plain_state(graph)
    original = SelfOrganization._commit_proposal

    def fail_after_commit(self, target_graph, node, proposal):
        original(self, target_graph, node, proposal)
        raise RuntimeError("late commit failure")

    monkeypatch.setattr(SelfOrganization, "_commit_proposal", fail_after_commit)
    with pytest.raises(RuntimeError, match="late commit"):
        SelfOrganization()(graph, 0, tau=0.1, collect_metrics=True)

    assert _plain_state(graph) == before
    assert graph.graph["_node_cache"] is cache
    assert tuple(cache.items()) == cache_items == ((0, cached_node),)


def test_after_monitor_error_restores_monitor_and_graph_state() -> None:
    class RejectingMonitor:
        def __init__(self) -> None:
            self.events: list[str] = []

        def before_operator(self, graph, node) -> None:
            self.events.append("before")

        def after_operator(self, graph, node, operator) -> None:
            self.events.append("after")
            raise RuntimeError("monitor rejection")

    graph = _graph()
    monitor = RejectingMonitor()
    graph.graph["integrity_monitor"] = monitor
    before = _plain_state(graph)

    with pytest.raises(RuntimeError, match="monitor rejection"):
        SelfOrganization()(graph, 0, tau=0.1)

    assert _plain_state(graph) == before
    assert graph.graph["integrity_monitor"] is monitor
    assert monitor.events == []


def test_late_metrics_error_restores_the_complete_graph(monkeypatch) -> None:
    graph = _graph()
    before = _plain_state(graph)

    def reject_metrics(self, target_graph, node, state_before):
        raise RuntimeError("metrics rejection")

    monkeypatch.setattr(SelfOrganization, "_collect_metrics", reject_metrics)
    with pytest.raises(RuntimeError, match="metrics rejection"):
        SelfOrganization()(graph, 0, tau=0.1, collect_metrics=True)

    assert _plain_state(graph) == before
    assert "_node_cache" not in graph.graph


def test_thol_does_not_read_or_mutate_neighbor_history() -> None:
    graph = _graph()
    graph.add_node(
        2,
        **{
            ALIAS_EPI[0]: 0.3,
            ALIAS_VF[0]: 1.0,
            ALIAS_DNFR[0]: 0.1,
            ALIAS_THETA[0]: 0.11,
            "epi_history": (0.1, 0.2, 0.3),
        },
    )
    graph.add_edge(0, 2)
    neighbor_before = deepcopy(dict(graph.nodes[2]))

    SelfOrganization()(graph, 0, tau=0.1)

    assert dict(graph.nodes[2]) == neighbor_before
    assert len(graph.nodes[0]["sub_epis"]) == 1


def test_existing_subnode_identity_must_be_consistent() -> None:
    graph = _graph(metabolic=False, propagation=False)
    graph.add_node("foreign", parent_node=99)
    graph.nodes[0]["sub_nodes"] = ["foreign"]
    before = _plain_state(graph)

    with pytest.raises(OperatorPreconditionError, match="parent identity"):
        SelfOrganization()(graph, 0, tau=0.1)

    assert _plain_state(graph) == before


def test_child_identifier_collision_never_overwrites_an_existing_node() -> None:
    graph = _graph(metabolic=False, propagation=False)
    graph.add_node("0_sub_0", marker="preexisting")

    SelfOrganization()(graph, 0, tau=0.1)

    assert graph.nodes["0_sub_0"] == {"marker": "preexisting"}
    assert graph.nodes[0]["sub_nodes"] == ["0_sub_1"]
    assert graph.nodes["0_sub_1"]["parent_node"] == 0


def test_thol_rejects_legacy_network_propagation_prewrite() -> None:
    graph = _graph(metabolic=True, propagation=True)
    before = _plain_state(graph)

    with pytest.raises(OperatorPreconditionError, match="use Resonance"):
        SelfOrganization()(graph, 0, tau=0.1)

    assert _plain_state(graph) == before


def test_thol_and_public_metabolism_share_one_amplitude_kernel() -> None:
    graph = _graph(metabolic=True, propagation=False)
    signals = capture_network_signals(graph, 0)
    expected = metabolize_signals_into_subepi(
        get_attr(graph.nodes[0], ALIAS_EPI),
        signals,
        get_attr(graph.nodes[0], ALIAS_D2EPI),
        gradient_weight=COUPLING_MODERATE,
        complexity_weight=COUPLING_GENTLE,
    )

    SelfOrganization()(graph, 0, tau=0.1)

    assert graph.nodes[0]["sub_epis"][-1]["epi"] == pytest.approx(expected)

def test_thol_rollback_restores_nested_runtime_cache_contents() -> None:
    class MutatingMonitor:
        def before_operator(self, graph, node) -> None:
            return None

        def after_operator(self, graph, node, operator) -> None:
            graph.graph["custom_cache"]["nested"]["value"] = 9
            raise RuntimeError("reject after nested cache mutation")

    graph = _graph()
    cache = {"nested": {"value": 1}}
    graph.graph["custom_cache"] = cache
    graph.graph["integrity_monitor"] = MutatingMonitor()

    with pytest.raises(RuntimeError, match="nested cache mutation"):
        SelfOrganization()(graph, 0, tau=0.1)

    assert graph.graph["custom_cache"] is cache
    assert cache == {"nested": {"value": 1}}

def test_legacy_collective_threshold_is_inert() -> None:
    """Amplitude alignment no longer impersonates the canonical U5 target."""

    graph = _graph()
    graph.graph["THOL_MIN_COLLECTIVE_COHERENCE"] = object()

    SelfOrganization()(graph, 0, tau=0.1)

    assert "_thol_subepi_amplitude_alignment" in graph.nodes[0]
    assert "_thol_collective_coherence" not in graph.nodes[0]
    assert "thol_coherence_warnings" not in graph.graph

def test_thol_refreshes_stale_acceleration_before_pressure_update() -> None:
    graph = _graph(metabolic=False, propagation=False)
    graph.nodes[0][ALIAS_D2EPI[0]] = True
    pressure_before = get_attr(graph.nodes[0], ALIAS_DNFR)
    expected_d2 = 0.4

    SelfOrganization()(graph, 0, tau=0.1)

    assert get_attr(graph.nodes[0], ALIAS_D2EPI) == pytest.approx(expected_d2)
    assert get_attr(graph.nodes[0], ALIAS_DNFR) == pytest.approx(
        pressure_before + COUPLING_GENTLE * expected_d2
    )


def test_thol_preserves_parent_and_neighbor_epi_coordinates() -> None:
    graph = _graph(metabolic=False, propagation=False)
    graph.nodes[0][ALIAS_EPI[0]] = 0.99
    graph.nodes[0]["epi_history"] = [0.0, 0.2, 0.99]
    graph.nodes[1][ALIAS_EPI[0]] = -0.95
    parent_before = get_attr(graph.nodes[0], ALIAS_EPI)
    neighbor_before = get_attr(graph.nodes[1], ALIAS_EPI)

    SelfOrganization()(graph, 0, tau=0.1)

    assert get_attr(graph.nodes[0], ALIAS_EPI) == parent_before
    assert get_attr(graph.nodes[1], ALIAS_EPI) == neighbor_before


def test_subepi_references_are_validated_even_with_materialized_children() -> None:
    graph = _graph(metabolic=False, propagation=False)
    graph.add_node(
        "valid-child",
        EPI=0.1,
        nu_f=0.5,
        delta_nfr=0.0,
        theta=0.1,
        parent_node=0,
        sub_epis=[],
        sub_nodes=[],
    )
    graph.nodes[0]["sub_nodes"] = ["valid-child"]
    graph.nodes[0]["sub_epis"] = [{"epi": 0.1, "node_id": "missing-child"}]
    before = _plain_state(graph)

    with pytest.raises(OperatorPreconditionError, match="node_id reference.*missing"):
        SelfOrganization()(graph, 0, tau=0.1)

    assert _plain_state(graph) == before


def test_thol_preserves_signed_acceleration_in_pressure_and_record() -> None:
    graph = _graph(metabolic=False, propagation=False)
    graph.nodes[0]["epi_history"] = [0.0, 1.0, 0.6]
    pressure_before = get_attr(graph.nodes[0], ALIAS_DNFR)
    expected_d2 = -1.4

    SelfOrganization()(graph, 0, tau=0.1)

    assert get_attr(graph.nodes[0], ALIAS_D2EPI) == pytest.approx(expected_d2)
    assert get_attr(graph.nodes[0], ALIAS_DNFR) == pytest.approx(
        pressure_before + COUPLING_GENTLE * expected_d2
    )
    assert graph.nodes[0]["sub_epis"][-1]["d2_epi"] == pytest.approx(expected_d2)


def test_thol_depth_limit_blocks_child_but_keeps_pressure_reorganization() -> None:
    graph = _graph(metabolic=False, propagation=False)
    graph.graph["THOL_MAX_BIFURCATION_DEPTH"] = 0
    pressure_before = get_attr(graph.nodes[0], ALIAS_DNFR)

    SelfOrganization()(graph, 0, tau=0.1)

    assert "sub_epis" not in graph.nodes[0]
    assert "sub_nodes" not in graph.nodes[0]
    assert get_attr(graph.nodes[0], ALIAS_DNFR) != pressure_before
    assert graph.nodes[0]["_thol_depth_limit_reached"] is True
    assert graph.graph["thol_depth_limits"] == [
        {"node": 0, "depth": 0, "max_depth": 0}
    ]
