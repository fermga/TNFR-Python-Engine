"""Cross-layer temporal-history precedence for legacy operator validators."""

from __future__ import annotations

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
from tnfr.operators.preconditions import (
    OperatorPreconditionError,
    validate_self_organization,
)
from tnfr.operators.preconditions.mutation import diagnose_mutation_readiness


def _graph(*, epi: float = 1.0) -> nx.Graph:
    graph = nx.Graph(
        ZHIR_THRESHOLD_XI=0.1,
        THOL_METABOLIC_ENABLED=False,
        BIFURCATION_THRESHOLD_TAU=0.1,
    )
    graph.add_node(
        0,
        **{
            ALIAS_EPI[0]: epi,
            ALIAS_VF[0]: 1.0,
            ALIAS_DNFR[0]: 0.2,
            ALIAS_THETA[0]: 0.0,
            "glyph_history": ["IL", "OZ"],
        },
    )
    graph.add_node(
        1,
        **{
            ALIAS_EPI[0]: 0.4,
            ALIAS_VF[0]: 1.0,
            ALIAS_DNFR[0]: 0.1,
            ALIAS_THETA[0]: 0.1,
        },
    )
    graph.add_edge(0, 1)
    return graph


def test_legacy_thol_validator_uses_authoritative_physical_acceleration() -> None:
    graph = _graph()
    graph.nodes[0]["epi_time_history"] = [(0.0, 0.0), (1.0, 0.1), (3.0, 1.0)]
    graph.nodes[0]["epi_history"] = [1.0, 1.0, 1.0]
    graph.nodes[0]["_epi_history"] = [0.0, 10.0, 0.0]

    validate_self_organization(graph, 0)

    assert graph.nodes[0]["_thol_no_bifurcation_expected"] is False
    assert ALIAS_D2EPI[0] not in graph.nodes[0]


def test_legacy_thol_validator_preserves_public_before_private_precedence() -> None:
    graph = _graph()
    graph.nodes[0]["epi_history"] = [0.0, 0.1, 1.0]
    graph.nodes[0]["_epi_history"] = [1.0, 1.0, 1.0]

    validate_self_organization(graph, 0)

    assert graph.nodes[0]["_thol_no_bifurcation_expected"] is False


def test_mutation_readiness_uses_physical_evidence_without_writes() -> None:
    graph = _graph(epi=0.4)
    graph.nodes[0]["epi_time_history"] = [(1.0, 0.1), (3.0, 0.4)]
    graph.nodes[0]["epi_history"] = [0.4, 0.4]
    graph.nodes[0]["_epi_history"] = [0.4, 1.0]
    before_node = deepcopy(dict(graph.nodes[0]))
    before_graph = deepcopy(dict(graph.graph))

    report = diagnose_mutation_readiness(graph, 0)

    assert report["ready"] is True
    assert report["checks"]["threshold_crossing"]["depi_dt"] == pytest.approx(0.15)
    assert report["checks"]["history_length"] == {
        "passed": True,
        "length": 2,
        "required": 2,
        "source": "epi_time_history",
    }
    assert dict(graph.nodes[0]) == before_node
    assert dict(graph.graph) == before_graph


def test_legacy_thol_validator_rejects_stale_physical_history_without_fallback() -> None:
    graph = _graph()
    graph.nodes[0]["epi_time_history"] = [(0.0, 0.0), (1.0, 0.1), (2.0, 0.9)]
    graph.nodes[0]["epi_history"] = [0.0, 0.1, 1.0]

    with pytest.raises(OperatorPreconditionError, match="final EPI must match"):
        validate_self_organization(graph, 0)
