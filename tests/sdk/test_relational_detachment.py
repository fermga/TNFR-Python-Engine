"""SDK delegates and exact nested evidence for hypothetical two-ring removal."""

import math
import pickle
from dataclasses import dataclass, replace
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.sdk import Network, RelationalExchangeModel, relational_report_to_dict
from tnfr.utils.io import json_dumps, json_loads

CYCLES = (tuple(range(5)), tuple(range(5, 10)))


@pytest.mark.parametrize(
    "method, owner_name, arguments",
    (
        (
            "relational_cycle_capture",
            "certify_relational_cycle_capture",
            {"cycle": ("declared",), "target_sector": -1},
        ),
        (
            "relational_detachment",
            "observe_relational_detachment",
            {"cycles": (("declared",),), "target_sector": -1},
        ),
    ),
)
def test_sdk_delegates_without_independent_admission(
    monkeypatch, method, owner_name, arguments
):
    from tnfr.physics import relational_capture as owner

    network, model = Network(nx.path_graph(2)), RelationalExchangeModel(1)
    calls, marker = [], object()

    def observe(graph, **kwargs):
        calls.append((graph, kwargs))
        return marker

    monkeypatch.setattr(owner, owner_name, observe)
    assert getattr(network, method)(model, **arguments) is marker
    assert calls == [(network.G, {"model": model, **arguments})]


@pytest.fixture(scope="module")
def detachment():
    graph = nx.disjoint_union(nx.cycle_graph(5), nx.cycle_graph(5))
    graph.add_edges_from(((0, 5), (1, 6)))
    for node in graph:
        graph.nodes[node].update(
            EPI=1 / 128 if node % 5 == 0 else 0.0,
            theta=2 * math.pi * (node % 5) / 5,
            nu_f=1.0,
        )

    def snapshot():
        return pickle.dumps(
            (graph.graph, tuple(graph.nodes(data=True)), tuple(graph.edges(data=True)))
        )

    before = snapshot()
    result = Network(graph).relational_detachment(
        RelationalExchangeModel(1), cycles=CYCLES
    )
    assert snapshot() == before
    assert result.capture_admitted
    return result


def test_export_separates_component_capture_event_budget_and_field_changes(detachment):
    report = relational_report_to_dict(detachment)
    assert report["report_type"] == "RelationalDetachmentObservation"
    data = report["report"]
    assert data["capture_admitted"] is True
    assert data["removed_bridges"] == [[0, 5], [1, 6]]
    assert len(data["before"]["nodes"]) == 10
    assert [len(row["field"]["nodes"]) for row in data["components"]] == [5, 5]
    assert [row["target_sector"] for row in data["components"]] == [1, 1]
    assert data["reset"]["storage_change"] == {"numerator": 0, "denominator": 1}
    assert any(detachment.phase_rate_change)
    residual = detachment.storage_reconciliation_residual
    assert residual == detachment.field_storage_change - detachment.reset.storage_change
    assert data["storage_reconciliation_residual"] == {
        "numerator": residual.numerator,
        "denominator": residual.denominator,
    }
    assert json_loads(json_dumps(report)) == report
    component = relational_report_to_dict(detachment.components[0])
    assert component["report_type"] == "RelationalCycleCaptureCertificate"
    assert component["report"] == data["components"][0]
    data["reset"]["storage_change"]["numerator"] = 99
    assert detachment.reset.storage_change == Q(0)


@pytest.mark.parametrize(
    "location", ("cycle", "component", "bridge", "reset", "reset_edge", "before_edge")
)
def test_export_validates_labels_inside_every_nested_owner(detachment, location):
    @dataclass(frozen=True)
    class OpaqueLabel:
        value: int

    label = OpaqueLabel(5)
    report = detachment
    if location in ("cycle", "component"):
        component = report.components[0]
        if location == "cycle":
            component = replace(component, cycle=(label,))
        else:
            component = replace(
                component, field=replace(component.field, nodes=(label,))
            )
        report = replace(report, components=(component, report.components[1]))
    elif location == "bridge":
        report = replace(report, removed_bridges=((label, 5),))
    elif location == "reset":
        reset = report.reset
        transport = replace(
            reset.transport_reset, after=replace(reset.after, nodes=(label,))
        )
        report = replace(report, reset=replace(reset, transport_reset=transport))
    elif location == "reset_edge":
        report = replace(report, reset=replace(report.reset, edges_after=((label, 5),)))
    else:
        report = replace(report, before=replace(report.before, edges=((label, 5),)))
    with pytest.raises(TypeError, match="node labels"):
        relational_report_to_dict(report)
