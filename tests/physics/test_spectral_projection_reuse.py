"""Spectral observations retain node alignment and sequential hook behavior."""

from __future__ import annotations

from dataclasses import replace

import networkx as nx
import numpy as np
import pytest

from tnfr.mathematics import spectral as spectral_math
from tnfr.physics import spectral_conservation as spectral
from tnfr.physics.conservation import ConservationSnapshot

_FIELDS = ("phi_s", "grad_phi", "k_phi", "j_phi", "j_dnfr")
_DIAGNOSTICS = ("balance", "ward", "energy", "sectors", "drift", "classification")


def _snapshot(nodes, offset=0.0):
    def field(values):
        # Snapshot insertion order need not equal the graph's basis-row order.
        return dict(reversed(list(zip(nodes, values))))

    phi = np.array([1.0, -2.0, 4.0]) + offset
    curvature = np.array([0.5, 2.0, -1.0]) - 0.25 * offset
    return ConservationSnapshot(
        charge_density=field(phi + curvature),
        phi_s=field(phi),
        k_phi=field(curvature),
        j_phi=field([0.25, 1.0, -0.5]),
        j_dnfr=field([-1.0, 0.5, 2.0]),
        grad_phi=field([1.5, 0.5, 3.0]),
        divergence=field(np.array([3.0, -1.0, 2.0]) - offset),
    )


def _graph(nodes=(2, 0, 1)):
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_weighted_edges_from(
        [(nodes[0], nodes[1], 1.0), (nodes[1], nodes[2], 2.0)]
    )
    return graph


def _call(name, before, after, graph):
    if name == "balance":
        return spectral.verify_spectral_conservation_balance(
            before, after, graph, dt=0.5
        )
    if name == "ward":
        return spectral.compute_spectral_ward_identity(before, after, "supplied", graph)
    if name == "energy":
        return spectral.compute_spectral_structural_energy(before, after, graph, dt=0.5)
    if name == "sectors":
        return spectral.decompose_spectral_sectors(graph, before)
    if name == "drift":
        return spectral.compute_spectral_energy_conservation(before, after, graph)
    return spectral.classify_spectral_modes(graph, before, threshold=0.2)


def _vector(snapshot, field, nodes):
    return np.array([getattr(snapshot, field)[node] for node in nodes])


@pytest.mark.parametrize("nodes", [(2, 0, 1), ("last", 2, ("middle",))])
@pytest.mark.parametrize("diagnostic", _DIAGNOSTICS)
def test_public_observations_use_graph_basis_order(nodes, diagnostic):
    graph = _graph(nodes)
    before, after = _snapshot(nodes), _snapshot(nodes, 0.75)
    _, basis = spectral_math.get_laplacian_spectrum(graph)
    result = _call(diagnostic, before, after, graph)

    def projected(snapshot, field):
        return basis.T @ _vector(snapshot, field, nodes)

    if diagnostic == "balance":
        spatial_residual = (
            _vector(after, "charge_density", nodes)
            - _vector(before, "charge_density", nodes)
        ) / 0.5 + 0.5 * (
            _vector(before, "divergence", nodes) + _vector(after, "divergence", nodes)
        )
        np.testing.assert_allclose(
            result.rho_spectrum_before, projected(before, "charge_density")
        )
        np.testing.assert_allclose(result.mode_sources, basis.T @ spatial_residual)
        np.testing.assert_allclose(
            result.mode_residuals, np.abs(basis.T @ spatial_residual)
        )
    elif diagnostic == "ward":
        np.testing.assert_allclose(
            result.delta_rho_spectrum,
            projected(after, "charge_density") - projected(before, "charge_density"),
        )
    elif diagnostic == "energy":
        expected_before = 0.5 * sum(projected(before, field) ** 2 for field in _FIELDS)
        expected_after = 0.5 * sum(projected(after, field) ** 2 for field in _FIELDS)
        np.testing.assert_allclose(result.mode_energies_before, expected_before)
        np.testing.assert_allclose(result.mode_energies_after, expected_after)
        np.testing.assert_allclose(
            result.mode_derivatives, (expected_after - expected_before) / 0.5
        )
    elif diagnostic == "sectors":
        np.testing.assert_allclose(result.phi_s_spectrum, projected(before, "phi_s"))
        np.testing.assert_allclose(result.k_phi_spectrum, projected(before, "k_phi"))
    elif diagnostic == "drift":
        # Parseval gives an independent spatial-energy oracle.
        expected = sum(
            float(_vector(before, field, nodes) @ _vector(before, field, nodes))
            for field in _FIELDS
        )
        assert result["total_energy_before"] == pytest.approx(expected)
    else:
        np.testing.assert_allclose(
            result["mode_transport_rates"], projected(before, "divergence")
        )


def test_existing_divergence_zero_mode_is_not_erased():
    graph = nx.cycle_graph(3)
    nodes = tuple(graph)
    snapshot = replace(_snapshot(nodes), divergence={node: 2.0 for node in nodes})
    result = spectral.verify_spectral_conservation_balance(snapshot, snapshot, graph)
    expected = result.eigenvectors.T @ np.full(3, 2.0)
    np.testing.assert_allclose(result.mode_sources, expected, atol=1e-14)
    assert abs(result.mode_sources[0]) == pytest.approx(2.0 * np.sqrt(3.0))
    classified = spectral.classify_spectral_modes(graph, snapshot, threshold=0.1)
    np.testing.assert_allclose(classified["mode_transport_rates"], expected, atol=1e-14)
    assert classified["mode_labels"][0] != "conserved"


@pytest.mark.parametrize("diagnostic", _DIAGNOSTICS)
@pytest.mark.parametrize("mismatch", ["missing", "extra", "substituted"])
def test_snapshot_support_must_match_graph_before_projection(
    monkeypatch, diagnostic, mismatch
):
    graph = _graph()
    snapshot = _snapshot(tuple(graph))
    field = (
        "charge_density"
        if diagnostic in ("balance", "ward")
        else "divergence" if diagnostic == "classification" else "phi_s"
    )
    values = dict(getattr(snapshot, field))
    if mismatch != "extra":
        values.pop(0)
    if mismatch != "missing":
        values["foreign"] = 1.0
    snapshot = replace(snapshot, **{field: values})

    def forbidden(*args, **kwargs):
        pytest.fail("invalid snapshot support reached a projection")

    monkeypatch.setattr(spectral, "gft", forbidden)
    with pytest.raises(ValueError, match=field + " must match the graph node support"):
        _call(diagnostic, snapshot, snapshot, graph)


@pytest.mark.parametrize(
    ("diagnostic", "field"),
    [("balance", "charge_density"), ("balance", "divergence")]
    + [("energy", field) for field in _FIELDS],
)
def test_after_snapshot_checks_every_consumed_map_before_projection(
    monkeypatch, diagnostic, field
):
    graph = _graph()
    before = _snapshot(tuple(graph))
    after = replace(before, **{field: {}})

    def forbidden(*args, **kwargs):
        pytest.fail("invalid after snapshot reached a projection")

    monkeypatch.setattr(spectral, "gft", forbidden)
    with pytest.raises(ValueError, match=field + " must match the graph node support"):
        _call(diagnostic, before, after, graph)


@pytest.mark.parametrize("diagnostic", _DIAGNOSTICS)
def test_unused_snapshot_maps_do_not_add_observation_requirements(diagnostic):
    graph = _graph()
    snapshot = _snapshot(tuple(graph))
    expected = _call(diagnostic, snapshot, snapshot, graph)
    field = (
        "charge_density"
        if diagnostic in ("energy", "drift", "sectors", "classification")
        else "j_phi"
    )
    incomplete = replace(snapshot, **{field: {"foreign": 9.0}})
    result = _call(diagnostic, incomplete, incomplete, graph)
    left = expected if isinstance(expected, dict) else vars(expected)
    right = result if isinstance(result, dict) else vars(result)
    for key, value in left.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, right[key])
        else:
            assert value == right[key]


@pytest.mark.parametrize("diagnostic", _DIAGNOSTICS[:-1])
def test_reused_projection_matches_sequential_public_transform(monkeypatch, diagnostic):
    graph = _graph()
    before, after = _snapshot(tuple(graph)), _snapshot(tuple(graph), 0.75)
    reused = _call(diagnostic, before, after, graph)
    canonical = spectral.gft
    observed = []

    def custom(signal, basis):
        observed.append(signal.copy())
        return canonical(signal, basis)

    monkeypatch.setattr(spectral, "gft", custom)
    sequential = _call(diagnostic, before, after, graph)
    expected_calls = {"balance": 4, "ward": 2, "energy": 10, "sectors": 2, "drift": 10}
    assert len(observed) == expected_calls[diagnostic]
    left = reused if isinstance(reused, dict) else vars(reused)
    right = sequential if isinstance(sequential, dict) else vars(sequential)
    for key, value in left.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, right[key])
        else:
            assert value == right[key]


def test_field_override_keeps_projection_and_energy_effects_interleaved(monkeypatch):
    graph = _graph()
    before, after = _snapshot(tuple(graph)), _snapshot(tuple(graph), 0.75)
    calls = []
    shared_result = np.zeros(3)

    def custom(signal, basis):
        calls.append(signal.copy())
        shared_result.fill(len(calls))
        return shared_result

    monkeypatch.setattr(spectral, "gft", custom)
    result = spectral.compute_spectral_structural_energy(before, after, graph)
    expected_calls = [
        _vector(snapshot, field, tuple(graph))
        for field in _FIELDS
        for snapshot in (before, after)
    ]
    np.testing.assert_array_equal(calls, expected_calls)
    # Each pair returns shared storage. The existing callback path consumes
    # that pair before the next callback overwrites its contents.
    np.testing.assert_array_equal(result.mode_energies_before, np.full(3, 110.0))
    np.testing.assert_array_equal(result.mode_energies_after, np.full(3, 110.0))


def test_replacing_math_transform_does_not_replace_consumers_bound_transform(
    monkeypatch,
):
    graph = _graph()
    snapshot = _snapshot(tuple(graph))
    expected = spectral.compute_spectral_structural_energy(snapshot, snapshot, graph)

    def unexpected(*args, **kwargs):
        pytest.fail(
            "the consumer's separately bound transform must remain authoritative"
        )

    monkeypatch.setattr(spectral_math, "gft", unexpected)
    result = spectral.compute_spectral_structural_energy(snapshot, snapshot, graph)
    np.testing.assert_array_equal(
        result.mode_energies_before, expected.mode_energies_before
    )


def test_generic_signal_arithmetic_keeps_consumers_bound_transform(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("generic arithmetic must not replace the consumer's bound GFT")

    class HookReplacingScalar:
        def __rmul__(self, value):
            monkeypatch.setattr(spectral_math, "gft", unexpected)
            return value * 2.0

    first = np.array([HookReplacingScalar(), 1.0, 3.0], dtype=object)
    second = np.array([4.0, 5.0, 6.0])
    result = spectral._project_vectors((first, second), np.eye(3))
    np.testing.assert_array_equal(result[0], [2.0, 1.0, 3.0])
    np.testing.assert_array_equal(result[1], second)


def test_later_calls_reobserve_changed_basis_and_snapshot():
    graph = _graph()
    snapshot = _snapshot(tuple(graph))
    first = spectral.decompose_spectral_sectors(graph, snapshot)
    graph[2][0]["weight"] = 5.0
    snapshot.phi_s[0] = -9.0
    _, basis = spectral_math.get_laplacian_spectrum(graph)
    second = spectral.decompose_spectral_sectors(graph, snapshot)
    expected = basis.T @ _vector(snapshot, "phi_s", tuple(graph))
    np.testing.assert_allclose(second.phi_s_spectrum, expected)
    assert not np.allclose(first.phi_s_spectrum, second.phi_s_spectrum)


def _large_observation(monkeypatch):
    graph = nx.path_graph(101)
    maps = {node: float(node + 1) for node in graph}
    snapshot = ConservationSnapshot(
        **{field: maps.copy() for field in ("charge_density", "divergence", *_FIELDS)}
    )
    monkeypatch.setattr(
        spectral, "get_laplacian_spectrum", lambda graph: (np.arange(101), np.eye(101))
    )
    return graph, snapshot


def test_large_basis_retains_bound_transform_after_backend_replaces_math_hook(
    monkeypatch,
):
    graph, snapshot = _large_observation(monkeypatch)
    backend_calls = []

    class Backend:
        supports_autodiff = False

    def replacement(*args, **kwargs):
        pytest.fail("backend mutation must not replace the consumer's bound GFT")

    def backend():
        backend_calls.append(None)
        monkeypatch.setattr(spectral_math, "gft", replacement)
        return Backend()

    monkeypatch.setattr(spectral_math, "get_backend", backend)
    result = spectral.decompose_spectral_sectors(graph, snapshot)
    assert len(backend_calls) == 2
    np.testing.assert_array_equal(result.phi_s_spectrum, np.arange(1.0, 102.0))
    np.testing.assert_array_equal(result.k_phi_spectrum, np.arange(1.0, 102.0))


def test_large_basis_consumes_shared_backend_output_before_next_field(monkeypatch):
    graph, snapshot = _large_observation(monkeypatch)

    class SharedBufferBackend:
        supports_autodiff = True

        def __init__(self):
            self.calls = 0
            self.output = np.zeros(101)

        def as_array(self, array):
            return np.asarray(array)

        def conjugate_transpose(self, array):
            return array.conj().T

        def matmul(self, left, right):
            self.calls += 1
            self.output.fill(self.calls)
            return self.output

        def to_numpy(self, array):
            return array

    backend = SharedBufferBackend()
    monkeypatch.setattr(spectral_math, "get_backend", lambda: backend)
    result = spectral.compute_spectral_structural_energy(snapshot, snapshot, graph)
    assert backend.calls == 10
    np.testing.assert_array_equal(result.mode_energies_before, np.full(101, 110.0))
    np.testing.assert_array_equal(result.mode_energies_after, np.full(101, 110.0))
