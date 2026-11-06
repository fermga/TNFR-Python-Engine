"""Unavailable observation owners must not resemble successful zero read-outs."""

import networkx as nx
import pytest

from tnfr.sdk import simple


@pytest.mark.parametrize(
    "method",
    (
        "telemetry",
        "tensor_invariants",
        "emergent_fields",
        "complex_field",
        "j_phi",
        "j_dnfr",
    ),
)
def test_unavailable_field_owner_raises_even_for_empty_graph(monkeypatch, method):
    network = simple.Network(nx.Graph())
    monkeypatch.setattr(simple, "_HAS_FIELDS", False)
    with pytest.raises(ImportError, match="Structural field observations"):
        getattr(network, method)()


@pytest.mark.parametrize(
    "method", ("structural_charge", "noether_charge", "candidate_energy", "energy")
)
def test_unavailable_balance_owner_never_fabricates_zero(monkeypatch, method):
    network = simple.Network(nx.Graph())
    monkeypatch.setattr(simple, "_HAS_CONSERVATION", False)
    with pytest.raises(ImportError, match="Structural balance observations"):
        getattr(network, method)()


def test_balance_report_keeps_its_explicit_unavailable_envelope(monkeypatch):
    network = simple.Network(nx.Graph())
    monkeypatch.setattr(simple, "_HAS_CONSERVATION", False)
    result = network.balance_alerts()
    assert result["sample_available"] is False
    assert result["diagnostic_scope"] == "conservation_module_unavailable"


def test_available_owner_can_report_actual_empty_graph_zero():
    network = simple.Network(nx.Graph())
    assert network.candidate_energy() == 0.0
    assert network.structural_charge() == 0.0
    assert network.j_phi() == {}
    assert network.j_dnfr() == {}
