"""Regression tests for the static spectral field decomposition."""

from __future__ import annotations

from dataclasses import asdict, replace

import networkx as nx
import numpy as np

from tnfr.physics.conservation import (
    ConservationSnapshot,
    SpectralConservation,
    compute_spectral_conservation,
)
from tnfr.physics.structural_diffusion import symmetric_normalized_laplacian


def _snapshot(divergence: dict[str, float]) -> ConservationSnapshot:
    zeros = {node: 0.0 for node in divergence}
    return ConservationSnapshot(
        charge_density={node: float(i + 1) for i, node in enumerate(divergence)},
        phi_s=zeros.copy(),
        k_phi=zeros.copy(),
        j_phi=zeros.copy(),
        j_dnfr=zeros.copy(),
        grad_phi=zeros.copy(),
        divergence=dict(divergence),
    )


def test_static_spectral_activity_projects_the_existing_divergence_once():
    graph = nx.Graph()
    graph.add_nodes_from(["z", "a", "m"])
    graph.add_weighted_edges_from([("z", "a", 1.0), ("a", "m", 2.0)])
    snapshot = _snapshot({"z": 1.0, "a": -2.0, "m": 0.5})

    nodes, laplacian = symmetric_normalized_laplacian(graph)
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    divergence = np.asarray([snapshot.divergence[node] for node in nodes])
    expected = np.abs(eigenvectors.T @ divergence)

    result = compute_spectral_conservation(graph, snapshot)

    assert result.node_order == tuple(nodes)
    assert np.allclose(result.modal_divergence_magnitude, expected)
    assert np.array_equal(result.conservation_by_mode,
                          result.modal_divergence_magnitude)
    # The retired implementation multiplied the already-computed divergence
    # by one more eigenvalue and therefore erased its zero-mode component.
    assert not np.allclose(expected, np.abs(eigenvalues * (eigenvectors.T @ divergence)))


def test_legacy_mode_count_is_an_explicit_alias_for_the_median_split():
    graph = nx.path_graph(["third", "first", "second"])
    result = compute_spectral_conservation(
        graph,
        _snapshot({"third": 0.2, "first": -0.1, "second": 0.4}),
    )

    median = float(np.median(result.modal_divergence_magnitude))
    expected = int(np.sum(result.modal_divergence_magnitude <= median + 1e-15))
    assert result.low_divergence_activity_modes == expected
    assert result.dominant_conservation_modes == expected


def test_spectral_certificate_preserves_stored_dataclass_api():
    activity = np.array([0.25, 0.5])
    certificate = SpectralConservation(
        eigenvalues=np.array([0.0, 1.0]),
        rho_spectrum=np.array([1.0, -1.0]),
        div_spectrum=np.array([0.25, -0.5]),
        conservation_by_mode=activity,
        dominant_conservation_modes=1,
        spectral_gap=1.0,
    )

    payload = asdict(certificate)
    assert np.array_equal(payload["conservation_by_mode"], activity)
    assert payload["dominant_conservation_modes"] == 1
    assert "modal_divergence_magnitude" not in payload
    assert "low_divergence_activity_modes" not in payload
    assert np.array_equal(certificate.modal_divergence_magnitude, activity)
    assert certificate.low_divergence_activity_modes == 1

    replaced = replace(
        certificate,
        conservation_by_mode=np.array([0.0, 0.0]),
        dominant_conservation_modes=2,
    )
    assert np.array_equal(replaced.modal_divergence_magnitude, [0.0, 0.0])
    assert replaced.low_divergence_activity_modes == 2
