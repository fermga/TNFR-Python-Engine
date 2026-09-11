"""Scoped certificates for observer-aware and composed graph reductions.

These utilities extend generator intertwining with measured observer defects
and static/dynamic controls. They do not certify U5 or the temporal REMESH
contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..alias import set_attr
from ..constants.aliases import (
    ALIAS_DEPI,
    ALIAS_DNFR,
    ALIAS_EPI,
    ALIAS_THETA,
    ALIAS_VF,
)
from ..metrics.common import compute_coherence, structural_coherence
from .canonical import (
    compute_phase_curvature,
    compute_phase_gradient,
    compute_structural_potential,
)
from .fields import classify_nodal_topology
from .spectral_projectors import matrix_exponential
from .structural_diffusion import structural_diffusion_operator
from .structural_morphism import (
    finite_time_intertwining_bound,
    intertwining_residual,
)
from .transient_u2 import potential_operator_from_graph

__all__ = [
    "ObserverTransportCertificate",
    "KronReductionCertificate",
    "ComposedReductionCertificate",
    "observer_transport_certificate",
    "kron_reduction_certificate",
    "composed_reduction_certificate",
]


@dataclass(frozen=True)
class ObserverTransportCertificate:
    """Canonical-observation defects under one state transport."""

    generator_kind: str
    generator_intertwining_residual: float
    state_transport_defect: float
    local_coherence_defect: float
    global_coherence_defect: float
    phase_gradient_defect: float
    phase_curvature_defect: float
    potential_defect: float
    potential_kernel_transport_residual: float
    source_topology: str
    target_topology: str
    topology_preserved: bool
    node_order_source: tuple[Any, ...]
    node_order_target: tuple[Any, ...]
    scope: str


def _configured_graph(
    graph: Any,
    nodes: tuple[Any, ...],
    state: np.ndarray,
    phase: np.ndarray,
    capacity: np.ndarray,
    laplacian: np.ndarray,
) -> Any:
    configured = graph.copy()
    pressure = -(laplacian @ state)
    rate = capacity * pressure
    for index, node in enumerate(nodes):
        data = configured.nodes[node]
        set_attr(data, ALIAS_EPI, float(state[index]))
        set_attr(data, ALIAS_THETA, float(phase[index]))
        set_attr(data, ALIAS_VF, float(capacity[index]))
        set_attr(data, ALIAS_DNFR, float(pressure[index]))
        set_attr(data, ALIAS_DEPI, float(rate[index]))
    return configured


def _field_vector(
    values: dict[Any, float], nodes: tuple[Any, ...]
) -> np.ndarray:
    return np.asarray([values[node] for node in nodes], dtype=float)


def observer_transport_certificate(
    source_graph: Any,
    target_graph: Any,
    morphism,
    source_state,
    target_state,
    source_phase,
    target_phase,
    source_capacity,
    target_capacity,
) -> ObserverTransportCertificate:
    """Measure which canonical observations survive a declared reduction.

    Graphs determine ``L_rw`` and the canonical potential distance kernels.
    Capacity is included in the full generator ``diag(nu_f)L_rw``.  Phase is
    observational here; no phase dynamics or operator trajectory is asserted.
    """
    source_nodes_list, source_laplacian = structural_diffusion_operator(
        source_graph
    )
    target_nodes_list, target_laplacian = structural_diffusion_operator(
        target_graph
    )
    source_nodes = tuple(source_nodes_list)
    target_nodes = tuple(target_nodes_list)
    map_matrix = np.asarray(morphism, dtype=float)
    source_state_array = np.asarray(source_state, dtype=float)
    target_state_array = np.asarray(target_state, dtype=float)
    source_phase_array = np.asarray(source_phase, dtype=float)
    target_phase_array = np.asarray(target_phase, dtype=float)
    source_capacity_array = np.asarray(source_capacity, dtype=float)
    target_capacity_array = np.asarray(target_capacity, dtype=float)
    expected_shape = (len(target_nodes), len(source_nodes))
    if map_matrix.shape != expected_shape:
        raise ValueError("morphism dimensions must match graph node orders")
    for value, size, name in (
        (source_state_array, len(source_nodes), "source_state"),
        (target_state_array, len(target_nodes), "target_state"),
        (source_phase_array, len(source_nodes), "source_phase"),
        (target_phase_array, len(target_nodes), "target_phase"),
        (source_capacity_array, len(source_nodes), "source_capacity"),
        (target_capacity_array, len(target_nodes), "target_capacity"),
    ):
        if value.shape != (size,) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name} must be finite and aligned to its graph")

    source = _configured_graph(
        source_graph, source_nodes, source_state_array, source_phase_array,
        source_capacity_array, source_laplacian,
    )
    target = _configured_graph(
        target_graph, target_nodes, target_state_array, target_phase_array,
        target_capacity_array, target_laplacian,
    )
    source_pressure = -(source_laplacian @ source_state_array)
    target_pressure = -(target_laplacian @ target_state_array)
    source_rate = source_capacity_array * source_pressure
    target_rate = target_capacity_array * target_pressure
    source_local = np.asarray([
        structural_coherence(source_pressure[index], source_rate[index])
        for index in range(len(source_nodes))
    ])
    target_local = np.asarray([
        structural_coherence(target_pressure[index], target_rate[index])
        for index in range(len(target_nodes))
    ])
    source_gradient = _field_vector(
        compute_phase_gradient(source), source_nodes
    )
    target_gradient = _field_vector(
        compute_phase_gradient(target), target_nodes
    )
    source_curvature = _field_vector(
        compute_phase_curvature(source), source_nodes
    )
    target_curvature = _field_vector(
        compute_phase_curvature(target), target_nodes
    )
    source_potential = _field_vector(
        compute_structural_potential(source), source_nodes
    )
    target_potential = _field_vector(
        compute_structural_potential(target), target_nodes
    )
    kernel_source_nodes, kernel_source = potential_operator_from_graph(
        source_graph
    )
    kernel_target_nodes, kernel_target = potential_operator_from_graph(
        target_graph
    )
    if tuple(kernel_source_nodes) != source_nodes:
        raise ValueError("source potential kernel node order changed")
    if tuple(kernel_target_nodes) != target_nodes:
        raise ValueError("target potential kernel node order changed")
    source_topology = classify_nodal_topology(source)["topology"]
    target_topology = classify_nodal_topology(target)["topology"]
    source_generator = source_capacity_array[:, None] * source_laplacian
    target_generator = target_capacity_array[:, None] * target_laplacian
    return ObserverTransportCertificate(
        generator_kind="full_capacity_weighted_random_walk_generator",
        generator_intertwining_residual=intertwining_residual(
            map_matrix, source_generator, target_generator
        ),
        state_transport_defect=float(
            np.linalg.norm(
                map_matrix @ source_state_array - target_state_array
            )
        ),
        local_coherence_defect=float(
            np.linalg.norm(map_matrix @ source_local - target_local)
        ),
        global_coherence_defect=abs(
            float(compute_coherence(source)) - float(compute_coherence(target))
        ),
        phase_gradient_defect=float(
            np.linalg.norm(map_matrix @ source_gradient - target_gradient)
        ),
        phase_curvature_defect=float(
            np.linalg.norm(map_matrix @ source_curvature - target_curvature)
        ),
        potential_defect=float(
            np.linalg.norm(map_matrix @ source_potential - target_potential)
        ),
        potential_kernel_transport_residual=float(
            np.linalg.norm(
                map_matrix @ kernel_source - kernel_target @ map_matrix, 2
            )
        ),
        source_topology=source_topology,
        target_topology=target_topology,
        topology_preserved=source_topology == target_topology,
        node_order_source=source_nodes,
        node_order_target=target_nodes,
        scope=(
            "one fixed graph pair/state/phase observation; nonlinear observer "
            "defects are measured and not bounded by generator intertwining"
        ),
    )


@dataclass(frozen=True)
class KronReductionCertificate:
    """Static resistance preservation versus exact eliminated dynamics."""

    reduced_laplacian: np.ndarray
    resistance_residual: float
    arbitrary_state_velocity_defect: float
    quasistatic_velocity_defect: float
    memory_kernel_initial_norm: float
    boundary_indices: tuple[int, ...]
    interior_indices: tuple[int, ...]
    scope: str


def _resistance_matrix(laplacian: np.ndarray) -> np.ndarray:
    inverse = np.linalg.pinv(laplacian, hermitian=True)
    diagonal = np.diag(inverse)
    return diagonal[:, None] + diagonal[None, :] - 2.0 * inverse


def kron_reduction_certificate(
    conductance_laplacian,
    boundary_indices,
    state,
) -> KronReductionCertificate:
    """Compare a Schur complement with the unreduced dynamic boundary rate."""
    laplacian = np.asarray(conductance_laplacian, dtype=float)
    x = np.asarray(state, dtype=float)
    if (
        laplacian.ndim != 2
        or laplacian.shape[0] != laplacian.shape[1]
        or x.shape != (laplacian.shape[0],)
        or not np.all(np.isfinite(laplacian))
        or not np.all(np.isfinite(x))
    ):
        raise ValueError(
            "laplacian and state must form one finite square system"
        )
    if not np.allclose(laplacian, laplacian.T, atol=1e-12, rtol=1e-10):
        raise ValueError(
            "Kron control requires a symmetric conductance Laplacian"
        )
    boundary = tuple(int(index) for index in boundary_indices)
    if len(set(boundary)) != len(boundary) or not boundary:
        raise ValueError("boundary indices must be unique and nonempty")
    interior = tuple(index for index in range(len(x)) if index not in boundary)
    if not interior:
        raise ValueError("Kron control requires at least one interior node")
    bb = laplacian[np.ix_(boundary, boundary)]
    bi = laplacian[np.ix_(boundary, interior)]
    ib = laplacian[np.ix_(interior, boundary)]
    ii = laplacian[np.ix_(interior, interior)]
    inverse_ii = np.linalg.inv(ii)
    reduced = bb - bi @ inverse_ii @ ib
    xb = x[list(boundary)]
    xi = x[list(interior)]
    full_rate = -(bb @ xb + bi @ xi)
    reduced_rate = -(reduced @ xb)
    quasistatic = -(inverse_ii @ ib @ xb)
    quasistatic_rate = -(bb @ xb + bi @ quasistatic)
    full_resistance = _resistance_matrix(laplacian)
    reduced_resistance = _resistance_matrix(reduced)
    boundary_resistance = full_resistance[np.ix_(boundary, boundary)]
    return KronReductionCertificate(
        reduced.copy(),
        float(
            np.linalg.norm(
                boundary_resistance - reduced_resistance, np.inf
            )
        ),
        float(np.linalg.norm(full_rate - reduced_rate)),
        float(np.linalg.norm(quasistatic_rate - reduced_rate)),
        float(np.linalg.norm(bi @ ib, 2)),
        boundary,
        interior,
        (
            "static symmetric Schur complement; the quasistatic manifold "
            "recovers its rate exactly, while the declared arbitrary interior "
            "state has a nonzero Markov-rate defect. The nonzero K(0)=L_bi "
            "L_ib term records the generic memory convolution in exact "
            "interior elimination; it is not by itself a universal failure "
            "claim for every initial condition"
        ),
    )


@dataclass(frozen=True)
class ComposedReductionCertificate:
    """Generator, trajectory and linear-observer bounds for two reductions."""

    first_residual: float
    second_residual: float
    composed_residual: float
    residual_composition_bound: float
    trajectory_defect: float
    trajectory_bound: float
    observer_defect: float
    observer_bound: float
    potential_kernel_composed_residual: float | None
    source_dimension: int
    middle_dimension: int
    target_dimension: int
    dense_cubic_cost_ratio: float
    scope: str


def composed_reduction_certificate(
    first_map,
    second_map,
    source_generator,
    middle_generator,
    target_generator,
    source_state,
    *,
    structural_time: float,
    target_observer,
    source_potential_kernel=None,
    target_potential_kernel=None,
) -> ComposedReductionCertificate:
    """Certify residual accumulation and one declared linear observation."""
    first = np.asarray(first_map, dtype=float)
    second = np.asarray(second_map, dtype=float)
    source = np.asarray(source_generator, dtype=float)
    middle = np.asarray(middle_generator, dtype=float)
    target = np.asarray(target_generator, dtype=float)
    state = np.asarray(source_state, dtype=float)
    observer = np.asarray(target_observer, dtype=float)
    composed = second @ first
    residual_first = first @ source - middle @ first
    residual_second = second @ middle - target @ second
    residual_composed = composed @ source - target @ composed
    component_bound = (
        np.linalg.norm(second, 2) * np.linalg.norm(residual_first, 2)
        + np.linalg.norm(residual_second, 2) * np.linalg.norm(first, 2)
    )
    trajectory_defect, trajectory_bound = finite_time_intertwining_bound(
        composed,
        source,
        target,
        state,
        structural_time=structural_time,
    )
    source_flow = matrix_exponential(-structural_time * source) @ state
    target_flow = matrix_exponential(
        -structural_time * target
    ) @ (composed @ state)
    state_difference = composed @ source_flow - target_flow
    if observer.ndim != 2 or observer.shape[1] != target.shape[0]:
        raise ValueError("target observer must align with target coordinates")
    observer_defect = float(np.linalg.norm(observer @ state_difference))
    observer_bound = float(np.linalg.norm(observer, 2) * trajectory_bound)
    kernel_residual = None
    if (
        source_potential_kernel is not None
        or target_potential_kernel is not None
    ):
        if source_potential_kernel is None or target_potential_kernel is None:
            raise ValueError(
                "both potential kernels must be supplied together"
            )
        source_kernel = np.asarray(source_potential_kernel, dtype=float)
        target_kernel = np.asarray(target_potential_kernel, dtype=float)
        if (
            source_kernel.shape != source.shape
            or target_kernel.shape != target.shape
        ):
            raise ValueError(
                "potential kernels must align with their generators"
            )
        # Generator intertwining does not imply distance-kernel intertwining;
        # a nonzero value is an observer-loss result, not a transport defect.
        kernel_residual = float(
            np.linalg.norm(
                composed @ source_kernel - target_kernel @ composed, 2
            )
        )
    source_dimension = source.shape[0]
    middle_dimension = middle.shape[0]
    target_dimension = target.shape[0]
    cost_ratio = float(
        source_dimension ** 3
        / max(middle_dimension ** 3 + target_dimension ** 3, 1)
    )
    return ComposedReductionCertificate(
        float(np.linalg.norm(residual_first, 2)),
        float(np.linalg.norm(residual_second, 2)),
        float(np.linalg.norm(residual_composed, 2)),
        float(component_bound),
        trajectory_defect,
        trajectory_bound,
        observer_defect,
        observer_bound,
        kernel_residual,
        source_dimension,
        middle_dimension,
        target_dimension,
        cost_ratio,
        (
            "finite linear generator/observer certificate; cubic ratio is a "
            "representation proxy, not a measured speedup"
        ),
    )
