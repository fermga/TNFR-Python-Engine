"""Structural EPI identity and operator provenance are independent channels."""

from __future__ import annotations

from copy import deepcopy

import networkx as nx
import pytest

from tnfr.config import EPI_KIND_PRIMARY, SOURCE_GLYPH_PRIMARY, get_aliases
from tnfr.constants import SOURCE_GLYPH_PRIMARY as EXPORTED_SOURCE_GLYPH_PRIMARY
from tnfr.constants.aliases import ALIAS_EPI_KIND, ALIAS_SOURCE_GLYPH
from tnfr.glyph_runtime import last_glyph
from tnfr.node import NodeNX
from tnfr.operators import Mutation, apply_glyph
from tnfr.operators.metrics_structural import mutation_metrics
from tnfr.operators.postconditions import OperatorContractViolation
from tnfr.operators.postconditions.mutation import verify_identity_preserved
from tnfr.operators.preconditions import OperatorPreconditionError
from tnfr.physics.integrity import MonitorMode, enable_integrity_monitor


def _graph() -> nx.Graph:
    graph = nx.path_graph(("target", "peer"))
    for index, node in enumerate(graph):
        graph.nodes[node].update(
            EPI=1.0,
            nu_f=1.0,
            delta_nfr=0.1,
            theta=0.2 * index,
            epi_history=[0.0, 0.3],
            glyph_history=["IL", "OZ"],
        )
    graph.nodes["target"]["EPI_kind"] = "wave_packet"
    return graph


def test_identity_and_provenance_aliases_are_disjoint() -> None:
    assert get_aliases("EPI_KIND") == ("EPI_kind", "epi_kind")
    assert get_aliases("SOURCE_GLYPH") == ("source_glyph", "last_glyph")
    assert ALIAS_EPI_KIND == ("EPI_kind", "epi_kind")
    assert ALIAS_SOURCE_GLYPH == ("source_glyph", "last_glyph")
    assert set(ALIAS_EPI_KIND).isdisjoint(ALIAS_SOURCE_GLYPH)
    assert EPI_KIND_PRIMARY == "EPI_kind"
    assert SOURCE_GLYPH_PRIMARY == "source_glyph"
    assert EXPORTED_SOURCE_GLYPH_PRIMARY == SOURCE_GLYPH_PRIMARY


def test_source_glyph_does_not_initialize_node_identity() -> None:
    graph = nx.Graph()
    graph.add_node("legacy", source_glyph="AL")
    node = NodeNX.from_graph(graph, "legacy")

    assert node.source_glyph == "AL"
    assert node.epi_kind == ""

    node.epi_kind = "cell"
    assert graph.nodes["legacy"]["EPI_kind"] == "cell"
    assert graph.nodes["legacy"]["source_glyph"] == "AL"


def test_legacy_last_glyph_is_provenance_on_node_adapter() -> None:
    graph = nx.Graph()
    graph.add_node("legacy", last_glyph="OZ")
    node = NodeNX.from_graph(graph, "legacy")

    assert node.source_glyph == "OZ"
    assert node.epi_kind == ""


def test_last_glyph_prefers_history_then_provenance_but_never_identity() -> None:
    assert last_glyph({"glyph_history": ["AL", "EN"], "source_glyph": "OZ"}) == "EN"
    assert last_glyph({"source_glyph": "RA"}) == "RA"
    assert last_glyph({"last_glyph": "IL"}) == "IL"
    assert last_glyph({"epi_kind": "ZHIR"}) is None


def test_zhir_identity_check_ignores_provenance_and_catches_glyph_kind() -> None:
    graph = _graph()
    before = "wave_packet"

    graph.nodes["target"]["source_glyph"] = "ZHIR"
    verify_identity_preserved(graph, "target", before)

    graph.nodes["target"]["EPI_kind"] = "ZHIR"
    with pytest.raises(OperatorContractViolation, match="Structural identity changed"):
        verify_identity_preserved(graph, "target", before)


def test_mutation_capture_and_metrics_read_canonical_identity_only() -> None:
    graph = _graph()
    graph.nodes["target"]["source_glyph"] = "OZ"
    mutation = Mutation()

    captured = mutation._capture_state(graph, "target")
    assert captured["epi_kind"] == "wave_packet"

    graph.nodes["target"]["source_glyph"] = "ZHIR"
    metrics = mutation_metrics(
        graph,
        "target",
        theta_before=0.0,
        epi_before=1.0,
        vf_before=1.0,
        dnfr_before=0.1,
        epi_kind_before=captured["epi_kind"],
    )
    assert metrics["identity_preserved"] is True

    graph.nodes["target"]["EPI_kind"] = "ZHIR"
    metrics = mutation_metrics(
        graph,
        "target",
        theta_before=0.0,
        epi_before=1.0,
        epi_kind_before=captured["epi_kind"],
    )
    assert metrics["identity_preserved"] is False


def test_integrity_monitor_captures_identity_separately_from_provenance() -> None:
    graph = _graph()
    monitor = enable_integrity_monitor(graph, mode=MonitorMode.OBSERVE)

    monitor.before_operator(graph, "target")
    graph.nodes["target"]["theta"] = 0.4
    graph.nodes["target"]["source_glyph"] = "ZHIR"
    report = monitor.after_operator(graph, "target", "Mutation")
    assert report.postcondition_evaluated is True
    assert report.postcondition_ok is True

    monitor.before_operator(graph, "target")
    graph.nodes["target"]["theta"] = 0.8
    graph.nodes["target"]["EPI_kind"] = "ZHIR"
    report = monitor.after_operator(graph, "target", "Mutation")
    assert report.postcondition_evaluated is True
    assert report.postcondition_ok is False
    assert "Structural identity changed" in report.postcondition_detail


def test_mutation_threshold_is_a_non_disableable_atomic_gate() -> None:
    graph = _graph()
    graph.nodes["target"]["epi_history"] = [0.4, 0.4]
    graph.graph["VALIDATE_OPERATOR_PRECONDITIONS"] = False
    before = deepcopy(graph.nodes["target"])

    with pytest.raises(OperatorPreconditionError, match="signed dEPI/dt > xi"):
        Mutation()(graph, "target", validate_preconditions=False)

    assert graph.nodes["target"] == before


@pytest.mark.parametrize(
    "glyph",
    ["AL", "IL", "OZ", "UM", "SHA", "VAL", "NUL", "THOL", "ZHIR", "NAV", "REMESH"],
)
def test_dispatcher_records_provenance_without_retyping_structural_identity(
    glyph: str,
) -> None:
    graph = _graph()

    apply_glyph(graph, "target", glyph)

    assert graph.nodes["target"]["EPI_kind"] == "wave_packet"
    assert graph.nodes["target"]["source_glyph"] == glyph
    assert graph.nodes["target"]["glyph_history"][-1] == glyph
