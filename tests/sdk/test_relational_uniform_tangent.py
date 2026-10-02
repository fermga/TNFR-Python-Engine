"""SDK delegation and exact report projection for the shared tangent owner."""

from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_uniform_tangent,
)
from tnfr.sdk import Network, relational_report_to_dict


def _network(nodes=("a", "b")):
    graph = nx.Graph()
    graph.add_edge(*nodes)
    for node in graph:
        graph.nodes[node].update(EPI=0.0, theta=0.0, nu_f=1.0)
    return Network(graph)


def test_sdk_delegates_to_native_owner_and_exports_its_order_and_residuals():
    network = _network()
    model = RelationalExchangeModel(1.0)
    report = network.relational_uniform_tangent(model)
    assert report == evaluate_relational_uniform_tangent(network.G, model=model)
    payload = relational_report_to_dict(report)
    assert payload["report_type"] == "RelationalUniformTangent"
    assert payload["report"]["field"]["nodes"] == ["a", "b"]
    assert payload["report"]["generator"][0] == list(report.generator[0])
    for exact, encoded in zip(
        report.common_offset_residuals,
        payload["report"]["common_offset_residuals"],
        strict=True,
    ):
        assert tuple(Q(v["numerator"], v["denominator"]) for v in encoded) == exact


def test_tangent_export_keeps_node_label_admission():
    report = _network((object(), object())).relational_uniform_tangent(
        RelationalExchangeModel(1.0)
    )
    with pytest.raises(TypeError, match="node labels"):
        relational_report_to_dict(report)
