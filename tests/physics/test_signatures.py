"""Regression smoke tests for physics.signatures.

This module was previously broken by a cascade of pre-existing bugs (a cache
decorator called without its required ``level``/``dependencies`` arguments, an
invalid ``coherence_key`` kwarg passed to ``estimate_coherence_length``, and a
hard import of the removed ``examples_utils`` helper). These tests ensure the
module imports and both signature functions run, so the cascade cannot silently
reappear.
"""

from __future__ import annotations

from copy import deepcopy
import threading

import networkx as nx

from tnfr.physics.signatures import compute_au_like_signature, compute_element_signature


def _ring(n: int = 10) -> nx.Graph:
    G = nx.cycle_graph(n)
    for node in G.nodes():
        G.nodes[node]["delta_nfr"] = 0.1
        G.nodes[node]["theta"] = 0.0
        G.nodes[node]["EPI"] = 0.5
        G.nodes[node]["nu_f"] = 1.0
    return G


def test_compute_element_signature_runs():
    sig = compute_element_signature(_ring(), apply_synthetic_step=False)
    assert {
        "xi_c",
        "signature_class",
        "phi_s_drift",
        "synthetic_step_applied",
        "synthetic_probe_word",
        "signature_scope",
        "potential_drift_assessed",
    } <= set(sig)
    assert isinstance(sig["xi_c"], float)
    assert sig["synthetic_step_applied"] is False
    assert sig["synthetic_probe_word"] == ()
    assert sig["signature_scope"] == "unperturbed_snapshot"
    assert sig["potential_drift_assessed"] is False
    assert sig["potential_drift_ok"] is None
    assert not {
        "spectral_coherence",
        "phase_spectral_energy",
        "dnfr_spectral_energy",
    }.intersection(sig)


def test_synthetic_probe_is_real_and_leaves_input_graph_unchanged():
    graph = _ring()
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    before_graph = deepcopy(graph.graph)

    sig = compute_element_signature(graph, apply_synthetic_step=True)

    assert sig["synthetic_step_applied"] is True
    assert sig["synthetic_probe_word"] == ("emission", "coherence", "silence")
    assert sig["signature_scope"] == "finite_declared_probe_response"
    assert sig["potential_drift_assessed"] is True
    assert sig["phi_s_drift"] > 0.0
    assert dict(graph.nodes(data=True)) == before_nodes
    assert graph.graph == before_graph


def test_compute_au_like_signature_runs():
    au = compute_au_like_signature(_ring())
    assert isinstance(au, dict) and au


def test_signature_u6_drift_uses_nodal_max_without_spatial_cancellation():
    graph = nx.path_graph(2)
    for node, pressure in enumerate((0.5, -0.5)):
        graph.nodes[node].update(
            delta_nfr=pressure,
            theta=0.0,
            EPI=0.5,
            nu_f=1.0,
        )

    signature = compute_element_signature(graph, apply_synthetic_step=True)

    assert signature["phi_s_drift"] > 0.0
    assert signature["phi_s_mean_abs_drift"] > 0.0


def test_signature_probe_ignores_opaque_runtime_and_external_monitor():
    class RejectingMonitor:
        def __init__(self) -> None:
            self.events: list[str] = []

        def before_operator(self, graph, node) -> None:
            self.events.append("before")
            raise RuntimeError("probe must not call external monitor")

        def after_operator(self, graph, node, operator) -> None:
            self.events.append("after")

    graph = _ring()
    runtime_lock = threading.Lock()
    monitor = RejectingMonitor()
    graph.graph["runtime_lock"] = runtime_lock
    graph.graph["integrity_monitor"] = monitor
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    before_edges = deepcopy(tuple(graph.edges(data=True)))
    before_graph = dict(graph.graph)

    signature = compute_element_signature(graph, apply_synthetic_step=True)

    assert signature["synthetic_step_applied"] is True
    assert dict(graph.nodes(data=True)) == before_nodes
    assert tuple(graph.edges(data=True)) == before_edges
    assert graph.graph == before_graph
    assert graph.graph["runtime_lock"] is runtime_lock
    assert graph.graph["integrity_monitor"] is monitor
    assert monitor.events == []


def test_signature_cannot_be_stable_after_u6_drift_violation():
    graph = nx.path_graph(2)
    for node, pressure in enumerate((10.0, -10.0)):
        graph.nodes[node].update(
            delta_nfr=pressure,
            theta=0.0,
            EPI=0.5,
            nu_f=1.0,
        )

    signature = compute_element_signature(graph, apply_synthetic_step=True)

    assert signature["phi_s_drift"] > 0.5 * 3.141592653589793
    assert signature["potential_drift_ok"] is False
    assert signature["signature_class"] == "unstable"
