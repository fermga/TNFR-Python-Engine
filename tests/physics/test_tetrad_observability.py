"""Tetrad observability boundaries and capacity-aware readouts."""

from __future__ import annotations

import networkx as nx
import numpy as np

from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF
from tnfr.physics.fields import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
    estimate_coherence_length,
)
from tnfr.physics.observability import (
    finite_difference_observer_certificate,
    linear_observability_certificate,
    minimal_distinguishing_channels,
    observation_signature,
    observer_ablation_ranks,
    tetrad_observation_channels,
    transform_linear_observer,
)
from tnfr.physics.structural_diffusion import structural_diffusion_operator


def _state_with_uniform_capacity(capacity: float) -> nx.Graph:
    graph = nx.path_graph(4)
    for node, (epi, phase, pressure) in enumerate(
        ((0.0, 0.0, -1.0), (1.0, 0.2, 0.5), (-1.0, 0.4, 0.5), (0.5, 0.6, 0.0))
    ):
        graph.nodes[node][ALIAS_EPI[0]] = epi
        graph.nodes[node][ALIAS_THETA[0]] = phase
        graph.nodes[node][ALIAS_DNFR[0]] = pressure
        graph.nodes[node][ALIAS_VF[0]] = capacity
        graph.nodes[node]["glyph_history"] = ["AL", "IL", "SHA"]
    return graph


def _scalar_tetrad(graph: nx.Graph):
    return (
        compute_structural_potential(graph),
        compute_phase_gradient(graph),
        compute_phase_curvature(graph),
        estimate_coherence_length(graph),
    )


def _epi_velocity(graph: nx.Graph) -> np.ndarray:
    nodes, laplacian = structural_diffusion_operator(graph)
    epi = np.array([graph.nodes[node][ALIAS_EPI[0]] for node in nodes])
    capacity = np.array([graph.nodes[node][ALIAS_VF[0]] for node in nodes])
    return -capacity * (laplacian @ epi)


def test_tetrad_alone_does_not_identify_uniform_capacity_rate():
    slow = _state_with_uniform_capacity(1.0)
    fast = _state_with_uniform_capacity(2.0)
    slow_tetrad = _scalar_tetrad(slow)
    fast_tetrad = _scalar_tetrad(fast)
    for slow_value, fast_value in zip(slow_tetrad[:3], fast_tetrad[:3]):
        assert slow_value == fast_value
    assert slow_tetrad[3] == fast_tetrad[3]

    slow_velocity = _epi_velocity(slow)
    fast_velocity = _epi_velocity(fast)
    assert np.linalg.norm(slow_velocity) > 0.0
    assert np.allclose(
        fast_velocity, 2.0 * slow_velocity, atol=1e-12, rtol=1e-10
    )


def test_capacity_inclusive_observer_resolves_the_witness():
    slow = _state_with_uniform_capacity(1.0)
    fast = _state_with_uniform_capacity(2.0)
    assert [slow.nodes[node][ALIAS_VF[0]] for node in slow] == [1.0] * 4
    assert [fast.nodes[node][ALIAS_VF[0]] for node in fast] == [2.0] * 4
    assert [slow.nodes[node][ALIAS_DNFR[0]] for node in slow] == [
        fast.nodes[node][ALIAS_DNFR[0]] for node in fast
    ]
    assert [slow.nodes[node]["glyph_history"] for node in slow] == [
        fast.nodes[node]["glyph_history"] for node in fast
    ]

    for representation in ("global", "full"):
        tetrad_slow = observation_signature(
            slow,
            tetrad_representation=representation,
            include_pressure=True,
            include_history=True,
        )
        tetrad_fast = observation_signature(
            fast,
            tetrad_representation=representation,
            include_pressure=True,
            include_history=True,
        )
        assert tetrad_slow == tetrad_fast
        capacity_slow = observation_signature(
            slow,
            tetrad_representation=representation,
            include_capacity=True,
            include_pressure=True,
            include_history=True,
        )
        capacity_fast = observation_signature(
            fast,
            tetrad_representation=representation,
            include_capacity=True,
            include_pressure=True,
            include_history=True,
        )
        assert capacity_slow != capacity_fast
        assert minimal_distinguishing_channels(
            capacity_slow, capacity_fast
        ) == ("capacity",)


def test_tetrad_field_ablations_retain_capacity_counterexample():
    slow = _state_with_uniform_capacity(1.0)
    fast = _state_with_uniform_capacity(2.0)
    slow_fields = _scalar_tetrad(slow)
    fast_fields = _scalar_tetrad(fast)
    slow_velocity = _epi_velocity(slow)
    fast_velocity = _epi_velocity(fast)

    for omitted in range(4):
        retained = [index for index in range(4) if index != omitted]
        for index in retained[:3]:
            assert slow_fields[index] == fast_fields[index]
        if 3 in retained:
            assert slow_fields[3] == fast_fields[3]
        assert not np.allclose(slow_velocity, fast_velocity)


def _pressure_phase_observer(representation: str, omitted: str | None = None):
    def observe(state: np.ndarray) -> np.ndarray:
        graph = _state_with_uniform_capacity(1.0)
        for node in range(4):
            graph.nodes[node][ALIAS_DNFR[0]] = state[node]
            graph.nodes[node][ALIAS_THETA[0]] = state[4 + node]
        channels = tetrad_observation_channels(
            graph, representation=representation
        )
        return np.concatenate(
            [value for name, value in channels.items() if name != omitted]
        )

    return observe


def test_tetrad_local_rank_distinguishes_summary_and_fields():
    state = np.array([-1.0, 0.5, 0.5, 0.0, 0.0, 0.2, 0.4, 0.6])
    ranks = {}
    for representation in ("global", "full"):
        refined = []
        for step in (1e-4, 5e-5, 2.5e-5):
            certificate = finite_difference_observer_certificate(
                state,
                _pressure_phase_observer(representation),
                step=step,
                tolerance=1e-7,
                scope=f"path-4 {representation} tetrad branch",
            )
            refined.append(certificate.rank)
        assert len(set(refined)) == 1
        ranks[representation] = refined[0]
    assert ranks == {"global": 3, "full": 7}


def test_tetrad_field_ablation_ranks_are_recorded():
    state = np.array([-1.0, 0.5, 0.5, 0.0, 0.0, 0.2, 0.4, 0.6])
    expected = {
        "structural_potential": 3,
        "phase_gradient": 7,
        "phase_curvature": 7,
        "coherence_length": 7,
    }
    for omitted, expected_rank in expected.items():
        certificate = finite_difference_observer_certificate(
            state,
            _pressure_phase_observer("full", omitted),
            step=5e-5,
            tolerance=1e-7,
            scope=f"path-4 full tetrad without {omitted}",
        )
        assert certificate.rank == expected_rank


def test_global_mean_loses_nonuniform_epi_modes():
    graph = nx.path_graph(4)
    _, laplacian = structural_diffusion_operator(graph)
    generator = -laplacian
    degrees = np.array([graph.degree(node) for node in graph], dtype=float)
    global_mean = (degrees / degrees.sum())[None, :]
    full_field = np.eye(4)

    global_certificate = linear_observability_certificate(
        generator, global_mean, scope="global mean of path-4 EPI diffusion"
    )
    field_certificate = linear_observability_certificate(
        generator, full_field, scope="full path-4 EPI field"
    )

    assert global_certificate.rank == 1
    assert not global_certificate.is_state_observable
    assert field_certificate.rank == 4
    assert field_certificate.is_state_observable


def test_channel_ablation_has_fixed_rank_budget():
    generator = np.diag([0.0, -1.0, -2.0, -3.0])
    channels = {
        "global": np.array([[1.0, 0.0, 0.0, 0.0]]),
        "potential": np.array([[0.0, 1.0, 0.0, 0.0]]),
        "phase_gradient": np.array([[0.0, 0.0, 1.0, 0.0]]),
        "phase_curvature": np.array([[0.0, 0.0, 0.0, 1.0]]),
    }
    ranks = observer_ablation_ranks(generator, channels)
    assert ranks["all"] == 4
    assert all(ranks[f"without:{name}"] == 3 for name in channels)


def test_observability_rank_is_relabeling_invariant():
    graph = nx.path_graph(4)
    _, laplacian = structural_diffusion_operator(graph)
    generator = -laplacian
    observer = np.array([[1.0, 0.0, 0.0, 0.0]])
    permutation = np.eye(4)[[2, 0, 3, 1]]
    transformed = transform_linear_observer(generator, observer, permutation)
    original = linear_observability_certificate(generator, observer)
    relabelled = linear_observability_certificate(*transformed)
    assert relabelled.rank == original.rank
    assert np.allclose(relabelled.singular_values, original.singular_values)


def test_degenerate_basis_rotation_preserves_observability():
    generator = np.diag([0.0, -1.0, -1.0, -2.0])
    observer = np.array([[1.0, 1.0, 0.0, 1.0]])
    angle = 0.37
    rotation = np.eye(4)
    rotation[1:3, 1:3] = [
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ]
    transformed = transform_linear_observer(generator, observer, rotation)
    original = linear_observability_certificate(generator, observer)
    rotated = linear_observability_certificate(*transformed)
    assert rotated.rank == original.rank == 3
    assert np.allclose(rotated.singular_values, original.singular_values)
