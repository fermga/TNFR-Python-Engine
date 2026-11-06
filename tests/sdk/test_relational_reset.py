"""Public reset-budget delegation and exact export; no evolution certificate."""

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError
from fractions import Fraction as Q

import networkx as nx
import pytest

from tnfr.sdk import Network, export_to_json, relational_report_to_dict


def _networks():
    graph = nx.Graph()
    graph.add_edge(("port", 0), "right", weight=1.0)
    for node, form in zip(graph, (0.25, 0.75), strict=True):
        graph.nodes[node].update(EPI=form, theta=0.0, nu_f=1.0, delta_nfr=0.0)
    after = deepcopy(graph)
    after.nodes["right"]["EPI"] = 0.5
    after[("port", 0)]["right"]["weight"] = 2.0
    return Network(graph), Network(after)


def test_reset_sdk_delegates_both_graphs_and_scale_to_the_shared_owner(monkeypatch):
    from tnfr.physics import relational_observations as owner

    before, after = _networks()
    calls, marker = [], object()

    def observe(first, second, *, storage_scale):
        calls.append((first, second, storage_scale))
        return marker

    monkeypatch.setattr(owner, "observe_relational_reset", observe)
    assert before.relational_reset(after, storage_scale=1.5) is marker
    assert calls == [(before.G, after.G, 1.5)]


def test_reset_budget_has_exact_state_support_split_and_detached_json(tmp_path):
    before, after = _networks()
    originals = deepcopy(before.G), deepcopy(after.G)
    report = before.relational_reset(after, storage_scale=1.5)
    assert report.form_state_change == -Q(3, 32)
    assert report.form_support_change == Q(1, 32)
    assert report.form_storage_change == report.storage_change == -Q(1, 16)
    assert report.phase_storage_change == 0
    assert report.storage_before == Q(1, 8)
    assert report.storage_after == Q(1, 16)
    assert report.identity_residual == 0
    assert report.transport_reset.energy_change == report.form_support_change
    assert report.represented_zero_supply_passive
    with pytest.raises(FrozenInstanceError):
        report.storage_change = Q(0)
    assert nx.utils.graphs_equal(before.G, originals[0])
    assert nx.utils.graphs_equal(after.G, originals[1])

    data = relational_report_to_dict(report)
    assessment = report.assess_supply(Q(-1, 16))
    for name, payload in (
        ("reset", data),
        ("supply", relational_report_to_dict(assessment)),
    ):
        destination = tmp_path / f"{name}.json"
        export_to_json(payload, destination)
        assert json.loads(destination.read_text(encoding="utf-8")) == payload
    body = data["report"]
    assert data["schema"] == "tnfr.relational-report.v1"
    assert data["report_type"] == "RelationalResetObservation"
    assert body["before"]["nodes"] == [["port", 0], "right"]
    assert body["storage_change"] == {"numerator": -1, "denominator": 16}
    assert body["represented_zero_supply_passive"] is True
    assert assessment.represented_balance_satisfied and assessment.supply_margin == 0
    body["before"]["nodes"][0].append("mutated")
    body["storage_change"]["numerator"] = 99
    after.G.nodes["right"]["EPI"] = 12
    assert report.before.nodes == (("port", 0), "right")
    assert report.after.epi == (Q(1, 4), Q(1, 2))
    assert report.storage_change == -Q(1, 16)


def test_reset_sdk_requires_a_network_endpoint():
    before, after = _networks()
    with pytest.raises(TypeError, match="Network"):
        before.relational_reset(after.G, storage_scale=1)


def test_reset_export_rejects_opaque_labels_instead_of_stringifying_them():
    before, after = _networks()
    opaque = object()
    mapping = {"right": opaque}
    before = Network(nx.relabel_nodes(before.G, mapping))
    after = Network(nx.relabel_nodes(after.G, mapping))
    report = before.relational_reset(after, storage_scale=1)
    with pytest.raises(TypeError, match="node labels"):
        relational_report_to_dict(report)
