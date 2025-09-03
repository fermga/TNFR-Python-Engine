"""Pruebas de gamma."""

import math
import logging
import pytest

from tnfr.constants import attach_defaults, merge_overrides
from tnfr.dynamics import update_epi_via_nodal_equation
from tnfr.gamma import eval_gamma
from tnfr.helpers import increment_edge_version


def test_gamma_linear_integration(graph_canon):
    G = graph_canon()
    G.add_nodes_from([0, 1])
    attach_defaults(G)
    merge_overrides(G, GAMMA={"type": "kuramoto_linear", "beta": 1.0, "R0": 0.0})
    for n in G.nodes():
        G.nodes[n]["νf"] = 1.0
        G.nodes[n]["ΔNFR"] = 0.0
        G.nodes[n]["θ"] = 0.0
        G.nodes[n]["EPI"] = 0.0
    update_epi_via_nodal_equation(G, dt=1.0)
    assert pytest.approx(G.nodes[0]["EPI"], rel=1e-6) == 1.0
    assert pytest.approx(G.nodes[1]["EPI"], rel=1e-6) == 1.0


def test_gamma_bandpass_eval(graph_canon):
    G = graph_canon()
    G.add_nodes_from([0, 1, 2, 3])
    attach_defaults(G)
    merge_overrides(G, GAMMA={"type": "kuramoto_bandpass", "beta": 1.0})
    for n in [0, 1, 2]:
        G.nodes[n]["θ"] = 0.0
    G.nodes[3]["θ"] = math.pi
    g0 = eval_gamma(G, 0, t=0.0)
    g3 = eval_gamma(G, 3, t=0.0)
    assert pytest.approx(g0, rel=1e-6) == 0.25
    assert pytest.approx(g3, rel=1e-6) == -0.25


def test_gamma_tanh_eval(graph_canon):
    G = graph_canon()
    G.add_nodes_from([0, 1, 2, 3])
    attach_defaults(G)
    merge_overrides(
        G,
        GAMMA={"type": "kuramoto_tanh", "beta": 1.0, "k": 1.0, "R0": 0.0},
    )
    for n in [0, 1, 2]:
        G.nodes[n]["θ"] = 0.0
    G.nodes[3]["θ"] = math.pi
    expected = math.tanh(0.5)
    g0 = eval_gamma(G, 0, t=0.0)
    g3 = eval_gamma(G, 3, t=0.0)
    assert pytest.approx(g0, rel=1e-6) == expected
    assert pytest.approx(g3, rel=1e-6) == -expected


def test_gamma_harmonic_eval(graph_canon):
    G = graph_canon()
    G.add_nodes_from([0, 1])
    attach_defaults(G)
    merge_overrides(
        G, GAMMA={"type": "harmonic", "beta": 1.0, "omega": 1.0, "phi": 0.0}
    )
    for n in G.nodes():
        G.nodes[n]["θ"] = 0.0
    g0 = eval_gamma(G, 0, t=math.pi / 2)
    g1 = eval_gamma(G, 1, t=math.pi / 2)
    assert pytest.approx(g0, rel=1e-6) == 1.0
    assert pytest.approx(g1, rel=1e-6) == 1.0


def test_kuramoto_cache_invalidation_on_version(graph_canon):
    G = graph_canon()
    G.add_nodes_from([0, 1])
    attach_defaults(G)
    merge_overrides(G, GAMMA={"type": "kuramoto_linear", "beta": 1.0, "R0": 0.0})
    for n in G.nodes():
        G.nodes[n]["θ"] = 0.0
    g_before = eval_gamma(G, 0, t=0.0)

    G.add_node(2)
    G.nodes[2]["θ"] = math.pi
    increment_edge_version(G)
    g_after = eval_gamma(G, 0, t=0.0)

    assert g_after != pytest.approx(g_before)


def test_eval_gamma_logs_and_strict_mode(graph_canon, caplog):
    G = graph_canon()
    G.add_nodes_from([0])
    attach_defaults(G)
    merge_overrides(G, GAMMA={"type": "kuramoto_linear", "beta": "bad"})

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        g = eval_gamma(G, 0, t=0.0)
    assert g == 0.0
    assert any("Fallo al evaluar" in rec.message for rec in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        eval_gamma(G, 0, t=0.0, log_level=logging.WARNING)
    assert any("Fallo al evaluar" in rec.message for rec in caplog.records)

    with pytest.raises(ValueError):
        eval_gamma(G, 0, t=0.0, strict=True)


def test_eval_gamma_unknown_type_warning_and_strict(graph_canon, caplog):
    G = graph_canon()
    G.add_nodes_from([0])
    attach_defaults(G)
    merge_overrides(G, GAMMA={"type": "unknown_type"})

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        g = eval_gamma(G, 0, t=0.0)
    assert g == 0.0
    assert any("Tipo GAMMA desconocido" in rec.message for rec in caplog.records)

    with pytest.raises(ValueError):
        eval_gamma(G, 0, t=0.0, strict=True)
