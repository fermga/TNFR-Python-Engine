"""Pruebas de gamma."""

import math
import logging
import pytest

from tnfr.constants import inject_defaults, merge_overrides
from tnfr.dynamics import update_epi_via_nodal_equation
from tnfr.gamma import eval_gamma, GAMMA_REGISTRY, GammaEntry
from tnfr.utils.cache import EdgeCacheManager, increment_edge_version


def test_gamma_linear_integration(graph_canon):
    G = graph_canon()
    G.add_nodes_from([0, 1])
    inject_defaults(G)
    merge_overrides(
        G, GAMMA={"type": "kuramoto_linear", "beta": 1.0, "R0": 0.0}
    )
    for n in G.nodes():
        G.nodes[n]["νf"] = 1.0
        G.nodes[n]["ΔNFR"] = 0.0
        G.nodes[n]["θ"] = 0.0
        G.nodes[n]["EPI"] = 0.0
    update_epi_via_nodal_equation(G, dt=1.0)
    assert pytest.approx(G.nodes[0]["EPI"], rel=1e-6) == 1.0
    assert pytest.approx(G.nodes[1]["EPI"], rel=1e-6) == 1.0


def test_eval_gamma_none_returns_zero(graph_canon):
    G = graph_canon()
    G.add_node(0, θ=0.0)
    inject_defaults(G)
    G.graph["GAMMA"] = {"type": "none"}

    assert eval_gamma(G, 0, 1.0) == 0.0


def test_gamma_bandpass_eval(graph_canon):
    G = graph_canon()
    G.add_nodes_from([0, 1, 2, 3])
    inject_defaults(G)
    merge_overrides(G, GAMMA={"type": "kuramoto_bandpass", "beta": 1.0})
    for n in [0, 1, 2]:
        G.nodes[n]["θ"] = 0.0
    G.nodes[3]["θ"] = math.pi
    g0 = eval_gamma(G, 0, t=0.0)
    g3 = eval_gamma(G, 3, t=0.0)
    assert pytest.approx(g0, rel=1e-6) == 0.25
    assert pytest.approx(g3, rel=1e-6) == -0.25


def test_gamma_linear_string_params(graph_canon):
    G = graph_canon()
    G.add_nodes_from([0, 1])
    inject_defaults(G)
    merge_overrides(
        G, GAMMA={"type": "kuramoto_linear", "beta": "1.0", "R0": "0.0"}
    )
    for n in G.nodes():
        G.nodes[n]["θ"] = 0.0
    g0 = eval_gamma(G, 0, t=0.0)
    g1 = eval_gamma(G, 1, t=0.0)
    assert pytest.approx(g0, rel=1e-6) == 1.0
    assert pytest.approx(g1, rel=1e-6) == 1.0


def test_gamma_inplace_mutation_updates_spec(graph_canon):
    G = graph_canon()
    G.add_nodes_from([0, 1])
    inject_defaults(G)
    cfg = {"type": "kuramoto_linear", "beta": 1.0, "R0": 0.0}
    G.graph["GAMMA"] = cfg
    for n in G.nodes():
        G.nodes[n]["θ"] = 0.0

    g_initial = eval_gamma(G, 0, t=0.0)
    cfg["beta"] = 2.0
    g_updated = eval_gamma(G, 0, t=0.0)

    assert pytest.approx(g_initial, rel=1e-6) == 1.0
    assert pytest.approx(g_updated, rel=1e-6) == 2.0


def test_gamma_tanh_eval(graph_canon):
    G = graph_canon()
    G.add_nodes_from([0, 1, 2, 3])
    inject_defaults(G)
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


def test_gamma_bandpass_string_params(graph_canon):
    G = graph_canon()
    G.add_nodes_from([0, 1, 2, 3])
    inject_defaults(G)
    merge_overrides(G, GAMMA={"type": "kuramoto_bandpass", "beta": "1.0"})
    for n in [0, 1, 2]:
        G.nodes[n]["θ"] = 0.0
    G.nodes[3]["θ"] = math.pi
    g0 = eval_gamma(G, 0, t=0.0)
    g3 = eval_gamma(G, 3, t=0.0)
    assert pytest.approx(g0, rel=1e-6) == 0.25
    assert pytest.approx(g3, rel=1e-6) == -0.25


def test_gamma_harmonic_eval(graph_canon):
    G = graph_canon()
    G.add_nodes_from([0, 1])
    inject_defaults(G)
    merge_overrides(
        G, GAMMA={"type": "harmonic", "beta": 1.0, "omega": 1.0, "phi": 0.0}
    )
    for n in G.nodes():
        G.nodes[n]["θ"] = 0.0
    g0 = eval_gamma(G, 0, t=math.pi / 2)
    g1 = eval_gamma(G, 1, t=math.pi / 2)
    assert pytest.approx(g0, rel=1e-6) == 1.0
    assert pytest.approx(g1, rel=1e-6) == 1.0


def test_gamma_tanh_string_params(graph_canon):
    G = graph_canon()
    G.add_nodes_from([0, 1, 2, 3])
    inject_defaults(G)
    merge_overrides(
        G,
        GAMMA={
            "type": "kuramoto_tanh",
            "beta": "1.0",
            "k": "1.0",
            "R0": "0.0",
        },
    )
    for n in [0, 1, 2]:
        G.nodes[n]["θ"] = 0.0
    G.nodes[3]["θ"] = math.pi
    expected = math.tanh(0.5)
    g0 = eval_gamma(G, 0, t=0.0)
    g3 = eval_gamma(G, 3, t=0.0)
    assert pytest.approx(g0, rel=1e-6) == expected
    assert pytest.approx(g3, rel=1e-6) == -expected


def test_gamma_spec_normalized_once(graph_canon, monkeypatch):
    G = graph_canon()
    G.add_node(0, θ=0.0)
    G.graph["GAMMA"] = []  # invalid spec
    emitted = []

    def fake_warn(message, *_args, **_kwargs):
        emitted.append(message)

    monkeypatch.setattr("tnfr.graph_utils.warnings.warn", fake_warn)
    eval_gamma(G, 0, t=0.0)
    eval_gamma(G, 0, t=0.0)
    assert len(emitted) == 1


def test_default_gamma_spec_called_once(graph_canon, monkeypatch):
    from tnfr import gamma as gamma_mod

    G = graph_canon()
    G.graph.pop("GAMMA", None)
    G.add_node(0, θ=0.0)
    calls = []

    real = gamma_mod._default_gamma_spec

    def fake_default():
        calls.append(1)
        return real()

    monkeypatch.setattr(gamma_mod, "_default_gamma_spec", fake_default)
    gamma_mod.eval_gamma(G, 0, t=0.0)
    gamma_mod.eval_gamma(G, 0, t=0.0)
    assert calls == [1]


def test_kuramoto_cache_reuses_checksum(graph_canon, monkeypatch):
    from tnfr import gamma as gamma_mod

    G = graph_canon()
    G.add_node(0, θ=0.0)
    calls = []

    def fake_checksum(G):
        calls.append(1)
        return "sum"

    monkeypatch.setattr(gamma_mod, "node_set_checksum", fake_checksum)
    gamma_mod._ensure_kuramoto_cache(G, t=0)
    assert calls == [1]
    G.graph["_dnfr_nodes_checksum"] = "sum"
    gamma_mod._ensure_kuramoto_cache(G, t=1)
    assert calls == [1]


def test_kuramoto_cache_updates_on_time_change(graph_canon):
    from tnfr import gamma as gamma_mod

    G = graph_canon()
    G.add_nodes_from([0])
    inject_defaults(G)
    G.nodes[0]["θ"] = 0.0
    gamma_mod._ensure_kuramoto_cache(G, t=0)
    cache0 = G.graph["_kuramoto_cache"]
    gamma_mod._ensure_kuramoto_cache(G, t=1)
    cache1 = G.graph["_kuramoto_cache"]
    assert cache0 is not cache1


def test_kuramoto_cache_updates_on_nodes_change(graph_canon):
    from tnfr import gamma as gamma_mod

    G = graph_canon()
    G.add_nodes_from([0])
    inject_defaults(G)
    G.nodes[0]["θ"] = 0.0
    gamma_mod._ensure_kuramoto_cache(G, t=0)
    cache0 = G.graph["_kuramoto_cache"]
    G.add_node(1)
    G.nodes[1]["θ"] = 0.0
    gamma_mod._ensure_kuramoto_cache(G, t=0)
    cache1 = G.graph["_kuramoto_cache"]
    assert cache0 is not cache1


def test_kuramoto_cache_step_limit(graph_canon):
    from tnfr import gamma as gamma_mod

    G = graph_canon()
    G.add_nodes_from([0])
    inject_defaults(G)
    G.graph["KURAMOTO_CACHE_STEPS"] = 2
    G.nodes[0]["θ"] = 0.0
    gamma_mod._ensure_kuramoto_cache(G, t=0)
    gamma_mod._ensure_kuramoto_cache(G, t=1)
    gamma_mod._ensure_kuramoto_cache(G, t=2)
    cache, _ = EdgeCacheManager(G.graph).get_cache(2)
    entries = [k for k in cache if isinstance(k, tuple)]
    assert len(entries) == 2


def test_kuramoto_cache_invalidation_on_version(graph_canon):
    G = graph_canon()
    G.add_nodes_from([0, 1])
    inject_defaults(G)
    merge_overrides(
        G, GAMMA={"type": "kuramoto_linear", "beta": 1.0, "R0": 0.0}
    )
    for n in G.nodes():
        G.nodes[n]["θ"] = 0.0
    g_before = eval_gamma(G, 0, t=0.0)

    G.add_node(2)
    G.nodes[2]["θ"] = math.pi
    increment_edge_version(G)
    g_after = eval_gamma(G, 0, t=0.0)

    assert g_after != pytest.approx(g_before)


def test_gamma_harmonic_string_params(graph_canon):
    G = graph_canon()
    G.add_nodes_from([0, 1])
    inject_defaults(G)
    merge_overrides(
        G,
        GAMMA={
            "type": "harmonic",
            "beta": "1.0",
            "omega": "1.0",
            "phi": "0.0",
        },
    )
    for n in G.nodes():
        G.nodes[n]["θ"] = 0.0
    g0 = eval_gamma(G, 0, t=math.pi / 2)
    g1 = eval_gamma(G, 1, t=math.pi / 2)
    assert pytest.approx(g0, rel=1e-6) == 1.0
    assert pytest.approx(g1, rel=1e-6) == 1.0


def test_eval_gamma_logs_and_strict_mode(graph_canon, caplog):
    G = graph_canon()
    G.add_nodes_from([0])
    inject_defaults(G)
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


def test_eval_gamma_non_mapping_warns(graph_canon):
    G = graph_canon()
    G.add_nodes_from([0])
    inject_defaults(G)
    G.graph["GAMMA"] = "not a dict"
    with pytest.warns(UserWarning):
        g = eval_gamma(G, 0, t=0.0)
    assert g == 0.0


def test_eval_gamma_unknown_type_warning_and_strict(graph_canon, caplog):
    G = graph_canon()
    G.add_nodes_from([0])
    inject_defaults(G)
    merge_overrides(G, GAMMA={"type": "unknown_type"})

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        g = eval_gamma(G, 0, t=0.0)
    assert g == 0.0
    assert any(
        "Tipo GAMMA desconocido" in rec.message for rec in caplog.records
    )

    with pytest.raises(ValueError):
        eval_gamma(G, 0, t=0.0, strict=True)


def test_eval_gamma_unhandled_exception_propagates(graph_canon):
    G = graph_canon()
    G.add_node(0)
    inject_defaults(G)
    merge_overrides(G, GAMMA={"type": "bad"})

    def bad_gamma(G, node, t, cfg):
        raise RuntimeError("boom")

    GAMMA_REGISTRY["bad"] = GammaEntry(bad_gamma, False)
    try:
        with pytest.raises(RuntimeError):
            eval_gamma(G, 0, t=0.0, strict=False)
        with pytest.raises(RuntimeError):
            eval_gamma(G, 0, t=0.0, strict=True)
    finally:
        GAMMA_REGISTRY.pop("bad", None)
