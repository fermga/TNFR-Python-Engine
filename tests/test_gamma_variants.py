"""Pruebas de variantes de Γ."""

import math

from tnfr.constants import inject_defaults
from tnfr.gamma import eval_gamma, kuramoto_R_psi


def _prepare_graph(graph_canon, thetas) -> tuple:
    G = graph_canon()
    G.add_nodes_from(range(len(thetas)))
    inject_defaults(G)
    for idx, theta in enumerate(thetas):
        G.nodes[idx]["θ"] = theta
    return G, G.nodes[0]["θ"]


def test_eval_gamma_none_returns_zero(graph_canon):
    G, _ = _prepare_graph(graph_canon, [0.0])
    G.graph["GAMMA"] = {"type": "none"}
    assert eval_gamma(G, 0, 1.0) == 0.0


def test_eval_gamma_kuramoto_linear_matches_formula(graph_canon):
    G, theta0 = _prepare_graph(graph_canon, [0.0, 0.0])
    cfg = {"type": "kuramoto_linear", "beta": 0.5, "R0": 0.2}
    G.graph["GAMMA"] = cfg
    R, psi = kuramoto_R_psi(G)
    expected = cfg["beta"] * (R - cfg["R0"]) * math.cos(theta0 - psi)
    assert math.isclose(eval_gamma(G, 0, 0.0), expected, rel_tol=1e-12)


def test_eval_gamma_kuramoto_bandpass_matches_formula(graph_canon):
    G, theta0 = _prepare_graph(graph_canon, [0.0, math.pi / 2])
    cfg = {"type": "kuramoto_bandpass", "beta": 1.3}
    G.graph["GAMMA"] = cfg
    R, psi = kuramoto_R_psi(G)
    sgn = 1.0 if math.cos(theta0 - psi) >= 0.0 else -1.0
    expected = cfg["beta"] * R * (1.0 - R) * sgn
    assert math.isclose(eval_gamma(G, 0, 0.0), expected, rel_tol=1e-12)


def test_eval_gamma_kuramoto_tanh_matches_formula(graph_canon):
    G, theta0 = _prepare_graph(graph_canon, [0.0, 0.0])
    cfg = {"type": "kuramoto_tanh", "beta": 0.75, "k": 2.0, "R0": 0.1}
    G.graph["GAMMA"] = cfg
    R, psi = kuramoto_R_psi(G)
    expected = cfg["beta"] * math.tanh(cfg["k"] * (R - cfg["R0"])) * math.cos(theta0 - psi)
    assert math.isclose(eval_gamma(G, 0, 0.0), expected, rel_tol=1e-12)


def test_eval_gamma_harmonic_matches_formula(graph_canon):
    G, theta0 = _prepare_graph(graph_canon, [0.0, math.pi / 2])
    cfg = {"type": "harmonic", "beta": 1.5, "omega": 2.0, "phi": 0.3}
    G.graph["GAMMA"] = cfg
    R, psi = kuramoto_R_psi(G)
    t = 0.25
    expected = cfg["beta"] * math.sin(cfg["omega"] * t + cfg["phi"]) * math.cos(theta0 - psi)
    assert math.isclose(eval_gamma(G, 0, t), expected, rel_tol=1e-12)
