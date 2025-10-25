"""Tests for canonical clamp enforcement in runtime helpers."""

from __future__ import annotations

import math
from collections.abc import MutableSequence

import pytest

from tnfr.alias import get_attr, get_theta_attr
from tnfr.dynamics.aliases import ALIAS_EPI, ALIAS_VF
from tnfr.dynamics.runtime import apply_canonical_clamps


def test_apply_canonical_clamps_clamps_and_logs(graph_canon):
    """Nodal EPI/νf values clamp to bounds and strict graphs log alerts."""

    G = graph_canon()
    node = 0
    G.add_node(node, psi=5.0, nu_f=-3.0, theta=math.pi + 0.05)
    G.graph["VALIDATORS_STRICT"] = True
    G.graph["THETA_WRAP"] = True

    apply_canonical_clamps(G.nodes[node], G, node)

    nd = G.nodes[node]
    assert get_attr(nd, ALIAS_EPI, 0.0) == pytest.approx(1.0)
    assert get_attr(nd, ALIAS_VF, 0.0) == pytest.approx(0.0)

    wrapped_theta = get_theta_attr(nd, 0.0)
    expected_theta = ((math.pi + 0.05 + math.pi) % (2 * math.pi)) - math.pi
    assert wrapped_theta == pytest.approx(expected_theta)
    assert -math.pi <= wrapped_theta < math.pi

    alerts = G.graph["history"]["clamp_alerts"]
    assert len(alerts) == 2

    first = alerts[0]
    assert first["node"] == node
    assert first["attr"] == "EPI"
    assert first["value"] == pytest.approx(5.0)

    second = alerts[1]
    assert second["node"] == node
    assert second["attr"] == "VF"
    assert second["value"] == pytest.approx(-3.0)


def test_apply_canonical_clamps_updates_mapping_without_graph():
    """When called without a graph the mapping itself is clamped and updated."""

    node_data = {"psi": -5.0, "nu_f": 5.0, "theta": -math.pi - 0.2}

    apply_canonical_clamps(node_data)

    assert get_attr(node_data, ALIAS_EPI, 0.0) == pytest.approx(-1.0)
    assert get_attr(node_data, ALIAS_VF, 0.0) == pytest.approx(1.0)

    wrapped_theta = get_theta_attr(node_data, 0.0)
    expected_theta = ((-math.pi - 0.2 + math.pi) % (2 * math.pi)) - math.pi
    assert wrapped_theta == pytest.approx(expected_theta)
    assert node_data["theta"] == pytest.approx(expected_theta)
    assert node_data["phase"] == pytest.approx(expected_theta)


def test_apply_canonical_clamps_respects_disabled_theta_wrap(graph_canon):
    """θ remains unchanged when wrapping is disabled while other clamps persist."""

    G = graph_canon()
    node = 0
    original_theta = 1.5 * math.pi
    G.add_node(
        node,
        psi=5.0,
        nu_f=-3.0,
        theta=original_theta,
        phase=original_theta,
    )
    G.graph["VALIDATORS_STRICT"] = True
    G.graph["THETA_WRAP"] = False

    apply_canonical_clamps(G.nodes[node], G, node)

    nd = G.nodes[node]
    assert get_attr(nd, ALIAS_EPI, 0.0) == pytest.approx(1.0)
    assert get_attr(nd, ALIAS_VF, 0.0) == pytest.approx(0.0)
    assert get_theta_attr(nd, 0.0) == pytest.approx(original_theta)
    assert nd["theta"] == pytest.approx(original_theta)
    assert nd["phase"] == pytest.approx(original_theta)

    alerts = G.graph["history"]["clamp_alerts"]
    assert len(alerts) == 2
    assert all(alert["attr"] != "THETA" for alert in alerts)


def test_apply_canonical_clamps_rehydrates_tuple_alerts(graph_canon):
    """Strict graphs convert tuple clamp alerts into mutable sequences."""

    G = graph_canon()
    node = 1
    G.add_node(node, psi=2.5, nu_f=-1.5, theta=0.0)
    G.graph["VALIDATORS_STRICT"] = True
    history = G.graph.setdefault("history", {})
    history["clamp_alerts"] = ()

    apply_canonical_clamps(G.nodes[node], G, node)

    alerts = history["clamp_alerts"]
    assert isinstance(alerts, MutableSequence)
    assert len(alerts) == 2
    attrs = {alert["attr"] for alert in alerts}
    assert attrs == {"EPI", "VF"}
