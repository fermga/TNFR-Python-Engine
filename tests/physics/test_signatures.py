"""Regression smoke tests for physics.signatures.

This module was previously broken by a cascade of pre-existing bugs (a cache
decorator called without its required ``level``/``dependencies`` arguments, an
invalid ``coherence_key`` kwarg passed to ``estimate_coherence_length``, and a
hard import of the removed ``examples_utils`` helper). These tests ensure the
module imports and both signature functions run, so the cascade cannot silently
reappear.
"""

from __future__ import annotations

import math
import threading
from copy import deepcopy

import networkx as nx
import numpy as np
import pytest

from tnfr.physics.fields import estimate_coherence_length
from tnfr.physics.signatures import compute_au_like_signature, compute_element_signature
from tnfr.physics.structural_diffusion import structural_eigenvalues


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


def test_unperturbed_signature_does_not_install_cache_or_copy_opaque_handles():
    graph = _ring()
    runtime_lock = threading.Lock()
    graph.graph["runtime_lock"] = runtime_lock
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    before_graph = dict(graph.graph)

    signature = compute_element_signature(graph, apply_synthetic_step=False)

    assert signature["synthetic_step_applied"] is False
    assert signature["signature_scope"] == "unperturbed_snapshot"
    assert dict(graph.nodes(data=True)) == before_nodes
    assert graph.graph == before_graph
    assert graph.graph["runtime_lock"] is runtime_lock


@pytest.mark.parametrize("change_support", [False, True])
def test_signature_preserves_existing_spectrum_cache_identity_and_contents(
    change_support,
):
    graph = _ring()
    structural_eigenvalues(graph)
    original_cache = graph.graph["_tnfr_spectrum_cache"]
    original_signature = original_cache["sig"]
    original_values = original_cache["vals"].copy()
    marker = object()
    original_cache["caller_marker"] = marker
    if change_support:
        graph.edges[0, 1]["weight"] = 2.0
    before_nodes = deepcopy(dict(graph.nodes(data=True)))
    before_edges = deepcopy(tuple(graph.edges(data=True)))
    expected_length = estimate_coherence_length(graph.copy())

    signature = compute_element_signature(graph, apply_synthetic_step=False)

    assert signature["xi_c"] == expected_length
    assert signature["synthetic_step_applied"] is False
    assert graph.graph["_tnfr_spectrum_cache"] is original_cache
    assert set(original_cache) == {"sig", "vals", "caller_marker"}
    assert original_cache["sig"] == original_signature
    assert original_cache["caller_marker"] is marker
    np.testing.assert_array_equal(original_cache["vals"], original_values)
    assert dict(graph.nodes(data=True)) == before_nodes
    assert tuple(graph.edges(data=True)) == before_edges


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


@pytest.mark.parametrize("graph", [nx.Graph(), nx.DiGraph([(0, 1)])])
def test_unavailable_coherence_is_not_a_medium_or_stable_signature(graph):
    signature = compute_element_signature(graph, apply_synthetic_step=False)
    assert math.isnan(signature["xi_c"])
    assert signature["xi_c_available"] is False
    assert signature["xi_c_provenance"]["method"] == "unavailable"
    assert signature["coherence_length_category"] == "unavailable"
    assert signature["coherence_length_reference_diameter"] is None
    assert signature["signature_available"] is False
    assert signature["signature_class"] == "unavailable"
    if not graph:
        assert math.isnan(signature["mean_phase_gradient"])
        assert math.isnan(signature["max_phase_curvature_abs"])
        assert signature["phase_gradient_ok"] is False
        assert signature["curvature_hotspots_ok"] is False


def test_empty_graph_cannot_claim_a_completed_probe():
    signature = compute_element_signature(nx.Graph(), apply_synthetic_step=True)
    assert signature["synthetic_step_applied"] is False
    assert signature["potential_drift_assessed"] is False
    assert signature["potential_drift_ok"] is None
    assert signature["signature_class"] == "unavailable"


def test_spectral_scale_retains_evidence_without_a_path_length_category():
    graph = _ring()
    signature = compute_element_signature(graph, apply_synthetic_step=False)
    assert signature["xi_c_available"] is True
    assert signature["xi_c_provenance"]["method"] == "spectral_gap"
    assert "dimensionless" in signature["xi_c_provenance"]["distance_weighting"]
    assert signature["coherence_length_category"] == "not_comparable"
    assert signature["coherence_length_reference_diameter"] is None
    assert signature["signature_available"] is False
    assert signature["signature_class"] == "unavailable"


@pytest.mark.parametrize("directed", [False, True])
def test_signature_fit_categories_respect_actual_distance_geometry_and_units(directed):
    # C(center)=1 and C(leaf)=exp(-radius/2), giving exactly the
    # exponential product exp(-distance/2) in this star geometry.
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_node("center", delta_nfr=0.0)
    for leaf in range(12):
        radius = 1.0 + leaf % 4
        graph.add_edge("center", leaf, length=radius, weight=1.0)
        graph.nodes[leaf]["delta_nfr"] = math.expm1(radius / 2.0)
    first = compute_element_signature(graph, apply_synthetic_step=False)
    for _, _, data in graph.edges(data=True):
        data["length"] *= 3.0
    scaled = compute_element_signature(graph, apply_synthetic_step=False)
    expected_diameter = 4.0 if directed else 8.0
    assert first["xi_c_provenance"]["method"] == "autocorrelation_fit"
    assert first["xi_c"] == pytest.approx(2.0)
    assert scaled["xi_c"] == pytest.approx(6.0)
    assert first["coherence_length_reference_diameter"] == expected_diameter
    assert scaled["coherence_length_reference_diameter"] == 3 * expected_diameter
    assert first["coherence_length_category"] == ("medium" if directed else "localized")
    assert scaled["coherence_length_category"] == first["coherence_length_category"]
    assert first["signature_available"] is True
    assert first["signature_class"] == "stable"


def test_large_graph_size_does_not_override_unavailable_au_length(monkeypatch):
    from tnfr.operators import word_execution

    graph = nx.DiGraph((node, node + 1) for node in range(51))
    monkeypatch.setattr(word_execution, "run_network_sequence", lambda *a, **k: None)
    signature = compute_au_like_signature(graph)
    assert signature["xi_c_available"] is False
    assert signature["signature_available"] is False
    assert signature["is_au_like"] is False


def test_signature_mean_retains_representable_large_potential():
    graph = nx.path_graph(2)
    nx.set_node_attributes(graph, 1e308, "delta_nfr")
    signature = compute_element_signature(graph, apply_synthetic_step=False)
    assert signature["phi_s_before"] == 1e308
    assert signature["phi_s_after"] == 1e308
