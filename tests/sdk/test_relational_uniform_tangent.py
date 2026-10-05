"""SDK delegation and exact report projection for the shared tangent owner."""

from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.dynamics.relational import (
    RelationalExchangeModel,
    evaluate_relational_consensus_tangent,
    evaluate_relational_uniform_tangent,
)
from tnfr.sdk import Network, relational_report_to_dict


def _network(nodes=("a", "b")):
    graph = nx.Graph()
    graph.add_edge(*nodes)
    for node in graph:
        graph.nodes[node].update(EPI=0.0, theta=0.0, nu_f=1.0)
    return Network(graph)


@pytest.mark.parametrize(
    "method,owner,report_type",
    (
        (
            "relational_uniform_tangent",
            evaluate_relational_uniform_tangent,
            "RelationalUniformTangent",
        ),
        (
            "relational_consensus_tangent",
            evaluate_relational_consensus_tangent,
            "RelationalConsensusTangent",
        ),
    ),
)
def test_sdk_delegates_to_native_owner_and_exports_its_order_and_residuals(
    method, owner, report_type
):
    network = _network()
    if method == "relational_consensus_tangent":
        network.G.nodes["b"]["EPI"] = -0.5
    model = RelationalExchangeModel(1.0)
    report = getattr(network, method)(model)
    assert report == owner(network.G, model=model)
    payload = relational_report_to_dict(report)
    assert payload["report_type"] == report_type
    assert payload["report"]["field"]["nodes"] == ["a", "b"]
    assert payload["report"]["generator"][0] == list(report.generator[0])
    for exact, encoded in zip(
        report.common_offset_residuals,
        payload["report"]["common_offset_residuals"],
        strict=True,
    ):
        assert tuple(Q(v["numerator"], v["denominator"]) for v in encoded) == exact


@pytest.mark.parametrize(
    "method", ("relational_uniform_tangent", "relational_consensus_tangent")
)
def test_tangent_export_keeps_node_label_admission(method):
    report = getattr(_network((object(), object())), method)(
        RelationalExchangeModel(1.0)
    )
    with pytest.raises(TypeError, match="node labels"):
        relational_report_to_dict(report)


@pytest.mark.parametrize(
    "method", ("relational_uniform_tangent", "relational_consensus_tangent")
)
def test_tangent_sdk_has_no_parallel_admission(monkeypatch, method):
    from tnfr.dynamics import relational as owner

    network, model, calls, marker = _network(), object(), [], object()

    def observe(graph, **kwargs):
        calls.append((graph, kwargs))
        return marker

    monkeypatch.setattr(owner, "evaluate_" + method, observe)
    assert getattr(network, method)(model) is marker
    assert calls == [(network.G, {"model": model})]


def test_consensus_tangent_export_retains_moving_source_without_equilibrium_claim():
    network = _network()
    network.G.nodes["b"]["EPI"] = 1
    report = network.relational_consensus_tangent(RelationalExchangeModel(1))
    payload = relational_report_to_dict(report)
    assert any(payload["report"]["field"]["form_rate"])
    assert any(payload["report"]["field"]["phase_rate"])
    assert "target_sector" not in payload["report"]
    payload["report"]["field"]["epi"][0] = 999
    assert report.field.epi[0] == 0
