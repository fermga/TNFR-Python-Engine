"""Canonical word execution and read-only THOL metabolism compatibility."""

from __future__ import annotations

import math
from copy import deepcopy

import networkx as nx
import pytest

from tnfr.constants.aliases import (
    ALIAS_D2EPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_THETA,
    ALIAS_VF,
)
from tnfr.dynamics.metabolism import StructuralMetabolism
from tnfr.mathematics import BEPIElement
from tnfr.operators.definitions import SelfOrganization
from tnfr.operators.factor_contracts import GlyphFactorValidationError
from tnfr.operators.metabolism import (
    capture_network_signals,
    compose_subepi_amplitude,
    compute_cascade_depth,
    compute_hierarchical_depth,
    metabolize_signals_into_subepi,
    propagate_subepi_to_network,
)
from tnfr.operators.preconditions import OperatorPreconditionError


def _graph(*, propagation: bool = False) -> nx.Graph:
    graph = nx.Graph(
        RANDOM_SEED=17,
        NAV_RANDOM=False,
        OZ_ENABLE_PROPAGATION=False,
        THOL_METABOLIC_ENABLED=True,
        THOL_PROPAGATION_ENABLED=propagation,
        GLYPH_HYSTERESIS_WINDOW=30,
    )
    graph.add_node(
        0,
        **{
            ALIAS_EPI[0]: 0.6,
            ALIAS_VF[0]: 1.0,
            ALIAS_DNFR[0]: 0.2,
            ALIAS_THETA[0]: 0.1,
            ALIAS_D2EPI[0]: 0.2,
            "epi_history": [0.0, 0.1, 0.6],
            "glyph_history": [],
        },
    )
    graph.add_node(
        1,
        **{
            ALIAS_EPI[0]: 0.8,
            ALIAS_VF[0]: 1.0,
            ALIAS_DNFR[0]: 0.1,
            ALIAS_THETA[0]: 0.12,
            "epi_history": [0.6, 0.7, 0.8],
            "glyph_history": [],
        },
    )
    graph.add_edge(0, 1)
    return graph


def _plain_state(graph: nx.Graph) -> tuple[object, ...]:
    graph_data = {
        key: deepcopy(value)
        for key, value in graph.graph.items()
        if key not in {"_node_cache", "_node_cache_weak", "integrity_monitor"}
    }
    return (
        tuple(graph.nodes),
        tuple(graph.edges(data=True)),
        {node: deepcopy(dict(data)) for node, data in graph.nodes(data=True)},
        graph_data,
    )


def test_digest_executes_exact_contextualized_metabolic_word() -> None:
    graph = _graph()

    StructuralMetabolism(graph, 0).digest(tau=0.1)

    assert list(graph.nodes[0]["glyph_history"]) == [
        "EN",
        "OZ",
        "THOL",
        "IL",
        "NAV",
    ]


def test_moderate_adaptive_metabolism_does_not_substitute_away_thol() -> None:
    graph = _graph()

    StructuralMetabolism(graph, 0).adaptive_metabolism(0.1)

    assert list(graph.nodes[0]["glyph_history"]) == ["OZ", "THOL", "IL", "NAV"]


def test_each_cascade_level_has_its_own_oz_thol_il_context() -> None:
    graph = _graph()

    StructuralMetabolism(graph, 0).cascading_reorganization(depth=2)

    assert list(graph.nodes[0]["glyph_history"]) == [
        "OZ",
        "THOL",
        "IL",
        "OZ",
        "THOL",
        "IL",
        "NAV",
    ]


def test_late_closure_failure_rolls_back_the_complete_metabolic_word() -> None:
    graph = _graph()
    graph.graph["GLYPH_FACTORS"] = {"NAV_eta": "invalid"}
    before = _plain_state(graph)

    with pytest.raises(
        GlyphFactorValidationError, match="NAV_eta must be a finite real scalar"
    ):
        StructuralMetabolism(graph, 0).adaptive_metabolism(0.1)

    assert _plain_state(graph) == before
    assert "0_sub_0" not in graph
    assert "_node_cache" not in graph.graph


@pytest.mark.parametrize(
    ("method", "argument"),
    [
        ("digest", math.nan),
        ("adaptive_metabolism", -0.1),
        ("cascading_reorganization", True),
    ],
)
def test_invalid_public_inputs_are_rejected_without_state_changes(
    method: str, argument: object
) -> None:
    graph = _graph()
    before = _plain_state(graph)

    with pytest.raises(OperatorPreconditionError):
        getattr(StructuralMetabolism(graph, 0), method)(argument)

    assert _plain_state(graph) == before


def test_invalid_metabolic_rate_is_rejected_before_word_execution() -> None:
    graph = _graph()
    metabolism = StructuralMetabolism(graph, 0)
    metabolism.metabolic_rate = 0.0
    before = _plain_state(graph)

    with pytest.raises(OperatorPreconditionError, match="metabolic_rate"):
        metabolism.digest()

    assert _plain_state(graph) == before


def test_legacy_propagation_helper_rejects_unattached_records_without_writes() -> None:
    graph = _graph(propagation=True)
    record = {"epi": 0.2, "vf": 1.0, "timestamp": 1}
    before = _plain_state(graph)

    with pytest.raises(OperatorPreconditionError, match="not attached"):
        propagate_subepi_to_network(graph, 0, record)

    assert _plain_state(graph) == before


def test_legacy_propagation_helper_only_reads_an_attached_historical_event() -> None:
    graph = _graph(propagation=False)
    graph.nodes[0]["glyph_history"] = ["OZ"]
    SelfOrganization()(graph, 0, tau=0.1)
    record = graph.nodes[0]["sub_epis"][-1]
    expected = [(1, 0.125)]
    graph.graph["thol_propagations"] = [
        {
            "source_node": 0,
            "sub_epi": record["epi"],
            "timestamp": record["timestamp"],
            "propagations": list(expected),
        }
    ]
    before = _plain_state(graph)

    result = propagate_subepi_to_network(graph, 0, record)

    assert result == expected
    assert result is not graph.graph["thol_propagations"][-1]["propagations"]
    assert _plain_state(graph) == before


def test_capture_uses_circular_phase_variance_at_the_wrap_boundary() -> None:
    graph = _graph()
    graph.add_node(
        2,
        **{
            ALIAS_EPI[0]: 0.7,
            ALIAS_VF[0]: 1.0,
            ALIAS_DNFR[0]: 0.0,
            ALIAS_THETA[0]: math.tau - 0.01,
        },
    )
    graph.nodes[1][ALIAS_THETA[0]] = 0.01
    graph.add_edge(0, 2)

    signals = capture_network_signals(graph, 0)

    assert signals is not None
    assert signals["phase_variance"] < 1e-3


def test_capture_rejects_rich_epi_without_lossy_scalar_projection() -> None:
    graph = _graph()
    graph.nodes[1][ALIAS_EPI[0]] = BEPIElement(
        (0.4, 0.8), (0.4, 0.4), (0.0, 1.0)
    )
    before = _plain_state(graph)

    with pytest.raises(OperatorPreconditionError, match="signed scalar embedding"):
        capture_network_signals(graph, 0)

    assert _plain_state(graph) == before


def test_metabolize_helper_uses_shared_thol_scale_and_validates_signals() -> None:
    assert metabolize_signals_into_subepi(0.6, None, d2_epi=0.2) == pytest.approx(
        0.18
    )

    with pytest.raises(OperatorPreconditionError, match="circular phase variance"):
        metabolize_signals_into_subepi(
            0.6,
            {"epi_gradient": 0.1, "phase_variance": math.nan},
            d2_epi=0.2,
        )


def test_canonical_subepi_composition_has_no_inert_acceleration_channel() -> None:
    signals = {"epi_gradient": 0.1, "phase_variance": 0.2}

    canonical = compose_subepi_amplitude(0.6, signals)
    legacy_low = metabolize_signals_into_subepi(0.6, signals, d2_epi=-100.0)
    legacy_high = metabolize_signals_into_subepi(0.6, signals, d2_epi=100.0)

    assert canonical == pytest.approx(legacy_low)
    assert canonical == pytest.approx(legacy_high)


@pytest.mark.parametrize(
    "depth_function",
    [compute_cascade_depth, compute_hierarchical_depth],
)
def test_hierarchy_depth_rejects_parent_child_cycles(depth_function) -> None:
    graph = nx.DiGraph()
    graph.add_node("root", sub_nodes=[0], sub_epis=[])
    graph.add_node(0, sub_nodes=["root"], sub_epis=[])

    with pytest.raises(OperatorPreconditionError, match="contains a cycle"):
        depth_function(graph, "root")


def test_hierarchy_depth_traverses_falsy_child_identifier() -> None:
    graph = nx.DiGraph()
    graph.add_node(
        "root",
        sub_epis=[{"node_id": 0, "bifurcation_level": 1}],
        sub_nodes=[],
    )
    graph.add_node(
        0,
        sub_epis=[{"node_id": "leaf", "bifurcation_level": 1}],
        sub_nodes=[],
    )
    graph.add_node("leaf", sub_epis=[], sub_nodes=[])

    assert compute_cascade_depth(graph, "root") == 2
    assert compute_hierarchical_depth(graph, "root") == 2
