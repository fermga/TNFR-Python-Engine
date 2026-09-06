"""Observer-aware bounds for structural reductions."""

from __future__ import annotations

import networkx as nx
import numpy as np

from tnfr.mathematics.padic_tower import (
    compatible_connection_set,
    padic_laplacian,
    projective_scale_map,
)
from tnfr.physics.reduction_certificates import (
    composed_reduction_certificate,
    kron_reduction_certificate,
    observer_transport_certificate,
)
from tnfr.physics.spectral_projectors import matrix_exponential
from tnfr.physics.structural_diffusion import structural_diffusion_operator
from tnfr.physics.transient_u2 import potential_operator_from_graph


def _partition_map() -> np.ndarray:
    mapping = np.zeros((3, 6))
    for coarse, fiber in enumerate(((0, 3), (1, 4), (2, 5))):
        mapping[coarse, list(fiber)] = 0.5
    return mapping


def test_generator_transport_does_not_imply_all_observer_transport():
    source = nx.cycle_graph(6)
    target = nx.complete_graph(3)
    mapping = _partition_map()
    source_nodes, source_laplacian = structural_diffusion_operator(source)
    target_nodes, target_laplacian = structural_diffusion_operator(target)
    capacity_source = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    capacity_target = np.array([1.0, 2.0, 3.0])
    source_generator = capacity_source[:, None] * source_laplacian
    target_generator = capacity_target[:, None] * target_laplacian
    initial = np.array([1.0, -0.5, 0.25, 0.2, 1.5, -1.0])
    structural_time = 0.4
    source_state = matrix_exponential(
        -structural_time * source_generator
    ) @ initial
    target_state = matrix_exponential(
        -structural_time * target_generator
    ) @ (mapping @ initial)
    source_phase = np.array([0.0, 0.2, 0.4, 0.1, 0.3, 0.5])
    target_phase = mapping @ source_phase

    certificate = observer_transport_certificate(
        source,
        target,
        mapping,
        source_state,
        target_state,
        source_phase,
        target_phase,
        capacity_source,
        capacity_target,
    )
    assert tuple(source_nodes) == certificate.node_order_source
    assert tuple(target_nodes) == certificate.node_order_target
    assert certificate.generator_intertwining_residual < 1e-12
    assert certificate.state_transport_defect < 1e-12
    assert certificate.local_coherence_defect > 0.0
    assert certificate.global_coherence_defect > 0.0
    assert certificate.phase_gradient_defect < 1e-12
    assert certificate.phase_curvature_defect < 1e-12
    assert certificate.potential_defect > 0.0
    assert certificate.potential_kernel_transport_residual > 0.0
    assert certificate.topology_preserved


def test_kron_reduction_preserves_resistance_not_arbitrary_dynamics():
    laplacian = np.array(
        [[1.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]]
    )
    certificate = kron_reduction_certificate(
        laplacian, boundary_indices=(0, 2), state=[1.0, 2.0, -1.0]
    )
    assert certificate.resistance_residual < 1e-12
    assert certificate.arbitrary_state_velocity_defect > 0.0
    assert certificate.quasistatic_velocity_defect < 1e-12
    assert certificate.memory_kernel_initial_norm > 0.0


def _fraction_matrix(matrix) -> np.ndarray:
    return np.asarray([[float(value) for value in row] for row in matrix])


def _padic_graph(p: int, exponent: int, base: frozenset[int]) -> nx.DiGraph:
    size = p**exponent
    graph = nx.DiGraph()
    graph.add_nodes_from(range(size))
    for source in range(size):
        for shift in compatible_connection_set(p, exponent, base):
            graph.add_edge(source, (source + shift) % size, weight=1.0)
    return graph


def test_composed_reductions_bound_transport_and_observer_error():
    p = 3
    base = frozenset({1, 2})
    source = _fraction_matrix(
        padic_laplacian(p, 3, compatible_connection_set(p, 3, base))
    )
    middle = _fraction_matrix(
        padic_laplacian(p, 2, compatible_connection_set(p, 2, base))
    )
    target = _fraction_matrix(
        padic_laplacian(p, 1, compatible_connection_set(p, 1, base))
    )
    first = _fraction_matrix(projective_scale_map(p, 2))
    second = _fraction_matrix(projective_scale_map(p, 1))
    state = np.linspace(-1.0, 1.0, 27)
    source_graph = _padic_graph(p, 3, base)
    target_graph = _padic_graph(p, 1, base)
    _, source_kernel = potential_operator_from_graph(source_graph)
    _, target_kernel = potential_operator_from_graph(target_graph)
    observer = np.array([[1.0, 0.0, -1.0]])

    exact = composed_reduction_certificate(
        first,
        second,
        source,
        middle,
        target,
        state,
        structural_time=0.5,
        target_observer=observer,
        source_potential_kernel=source_kernel,
        target_potential_kernel=target_kernel,
    )
    assert exact.first_residual < 1e-12
    assert exact.second_residual < 1e-12
    assert exact.composed_residual < 1e-12
    assert exact.trajectory_defect < 1e-12
    assert exact.observer_defect < 1e-12
    assert exact.potential_kernel_composed_residual is not None
    assert exact.potential_kernel_composed_residual > 0.0
    assert exact.dense_cubic_cost_ratio > 1.0

    perturbed = first.copy()
    perturbed[0, 1] += 0.02
    measured = composed_reduction_certificate(
        perturbed,
        second,
        source,
        middle,
        target,
        state,
        structural_time=0.5,
        target_observer=observer,
    )
    assert measured.first_residual > 0.0
    assert (
        measured.composed_residual
        <= measured.residual_composition_bound + 1e-12
    )
    assert measured.trajectory_defect <= measured.trajectory_bound + 1e-12
    assert measured.observer_defect <= measured.observer_bound + 1e-12
