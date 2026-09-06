"""Scoped linear observability certificates for TNFR read-outs.

These helpers concern a declared finite-dimensional linear model ``x_dot=A x``
and observation ``y=C x``.  They do not assert that the canonical tetrad is a
complete state basis, nor do they linearize wrapped phase fields implicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

__all__ = [
    "LinearObservabilityCertificate",
    "LocalObserverCertificate",
    "ObservationSignature",
    "finite_difference_observer_certificate",
    "linear_observability_certificate",
    "minimal_distinguishing_channels",
    "observation_signature",
    "observer_ablation_ranks",
    "tetrad_observation_channels",
    "tetrad_observation_vector",
    "transform_linear_observer",
]


@dataclass(frozen=True)
class LinearObservabilityCertificate:
    """Rank certificate for a declared linear state/observer pair."""

    state_dimension: int
    observation_dimension: int
    rank: int
    tolerance: float
    singular_values: tuple[float, ...]
    is_state_observable: bool
    matrix: np.ndarray
    scope: str


@dataclass(frozen=True)
class LocalObserverCertificate:
    """Local finite-difference rank for a nonlinear observation map."""

    input_dimension: int
    output_dimension: int
    rank: int
    tolerance: float
    step: float
    singular_values: tuple[float, ...]
    jacobian: np.ndarray
    scope: str


@dataclass(frozen=True)
class ObservationSignature:
    """Explicit observation channels for a fixed state-comparison task."""

    tetrad_representation: str
    tetrad: tuple[float, ...]
    capacity: tuple[float, ...] | None
    pressure: tuple[float, ...] | None
    history: tuple[tuple[str, ...], ...] | None
    node_order: tuple[Any, ...]


def minimal_distinguishing_channels(
    left: ObservationSignature,
    right: ObservationSignature,
) -> tuple[str, ...]:
    """Return the smallest individual channels separating two signatures.

    This is task-specific pair discrimination, not Kalman state
    observability.  Categorical history remains categorical and capacity
    remains a model parameter rather than being forced into ``C x``.
    """
    if left.node_order != right.node_order:
        raise ValueError("signatures must use the same node order")
    channels = (
        ("tetrad", left.tetrad, right.tetrad),
        ("capacity", left.capacity, right.capacity),
        ("pressure", left.pressure, right.pressure),
        ("history", left.history, right.history),
    )
    return tuple(
        name
        for name, left_value, right_value in channels
        if left_value is not None
        and right_value is not None
        and left_value != right_value
    )


def _finite_matrix(value, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite two-dimensional matrix")
    return matrix


def tetrad_observation_vector(
    graph: Any,
    *,
    representation: str,
    node_order: Sequence[Any] | None = None,
) -> np.ndarray:
    """Return an actual global-summary or full-node tetrad observation.

    The full representation concatenates ``Phi_s``, ``|grad phi|`` and
    ``K_phi`` in the declared node order, followed by scalar ``xi_C``.  The
    global representation uses the mean of each nodal field plus ``xi_C``.
    """
    nodes = tuple(graph.nodes()) if node_order is None else tuple(node_order)
    if set(nodes) != set(graph.nodes()) or len(nodes) != len(graph):
        raise ValueError(
            "node_order must contain every graph node exactly once"
        )
    channels = tetrad_observation_channels(
        graph, representation=representation, node_order=nodes
    )
    return np.concatenate(tuple(channels.values()))


def tetrad_observation_channels(
    graph: Any,
    *,
    representation: str,
    node_order: Sequence[Any] | None = None,
) -> dict[str, np.ndarray]:
    """Return named tetrad channels under one declared representation."""
    from .canonical import (
        compute_phase_curvature,
        compute_phase_gradient,
        compute_structural_potential,
        estimate_coherence_length,
    )

    nodes = tuple(graph.nodes()) if node_order is None else tuple(node_order)
    if set(nodes) != set(graph.nodes()) or len(nodes) != len(graph):
        raise ValueError(
            "node_order must contain every graph node exactly once"
        )
    fields = {
        "structural_potential": compute_structural_potential(graph),
        "phase_gradient": compute_phase_gradient(graph),
        "phase_curvature": compute_phase_curvature(graph),
    }
    if representation == "global":
        channels = {
            name: np.asarray(
                [float(np.mean([field[node] for node in nodes]))], dtype=float
            )
            for name, field in fields.items()
        }
    elif representation == "full":
        channels = {
            name: np.asarray([field[node] for node in nodes], dtype=float)
            for name, field in fields.items()
        }
    else:
        raise ValueError("representation must be 'global' or 'full'")
    channels["coherence_length"] = np.asarray(
        [float(estimate_coherence_length(graph))], dtype=float
    )
    return channels


def observation_signature(
    graph: Any,
    *,
    tetrad_representation: str,
    include_capacity: bool = False,
    include_pressure: bool = False,
    include_history: bool = False,
    node_order: Sequence[Any] | None = None,
) -> ObservationSignature:
    """Build explicit channels without treating history as a numeric state."""
    from ..alias import get_attr
    from ..constants.aliases import ALIAS_DNFR, ALIAS_VF

    nodes = tuple(graph.nodes()) if node_order is None else tuple(node_order)
    tetrad = tuple(
        float(value) for value in tetrad_observation_vector(
            graph, representation=tetrad_representation, node_order=nodes
        )
    )
    capacity = (
        tuple(
            float(get_attr(graph.nodes[node], ALIAS_VF, 0.0))
            for node in nodes
        )
        if include_capacity else None
    )
    pressure = (
        tuple(
            float(get_attr(graph.nodes[node], ALIAS_DNFR, 0.0))
            for node in nodes
        )
        if include_pressure else None
    )
    history = (
        tuple(
            tuple(graph.nodes[node].get("glyph_history", ()))
            for node in nodes
        )
        if include_history else None
    )
    return ObservationSignature(
        tetrad_representation, tetrad, capacity, pressure, history, nodes
    )


def finite_difference_observer_certificate(
    state,
    observer: Callable[[np.ndarray], np.ndarray],
    *,
    step: float,
    tolerance: float | None = None,
    scope: str = "local nonlinear observer",
) -> LocalObserverCertificate:
    """Estimate a central-difference Jacobian and its local numerical rank."""
    point = np.asarray(state, dtype=float)
    if point.ndim != 1 or not np.all(np.isfinite(point)):
        raise ValueError("state must be a finite one-dimensional vector")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be finite and positive")
    baseline = np.asarray(observer(point.copy()), dtype=float)
    if baseline.ndim != 1 or not np.all(np.isfinite(baseline)):
        raise ValueError(
            "observer must return a finite one-dimensional vector"
        )
    jacobian = np.empty((baseline.size, point.size), dtype=float)
    for column in range(point.size):
        delta = np.zeros_like(point)
        delta[column] = step
        plus = np.asarray(observer(point + delta), dtype=float)
        minus = np.asarray(observer(point - delta), dtype=float)
        if plus.shape != baseline.shape or minus.shape != baseline.shape:
            raise ValueError("observer output shape must remain fixed")
        jacobian[:, column] = (plus - minus) / (2.0 * step)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    if tolerance is None:
        scale = singular_values[0] if singular_values.size else 0.0
        tolerance = max(jacobian.shape) * np.finfo(float).eps * scale / step
    rank = int(np.sum(singular_values > tolerance))
    return LocalObserverCertificate(
        point.size,
        baseline.size,
        rank,
        float(tolerance),
        float(step),
        tuple(float(value) for value in singular_values),
        jacobian.copy(),
        scope,
    )


def linear_observability_certificate(
    generator,
    observer,
    *,
    tolerance: float | None = None,
    scope: str = "declared linear model",
) -> LinearObservabilityCertificate:
    r"""Return the Kalman observability matrix ``[C; CA; ...; CA^(n-1)]``.

    The numerical rank uses a scale-aware SVD tolerance unless one is supplied.
    ``is_state_observable`` refers only to the declared coordinates and model.
    """
    a = _finite_matrix(generator, "generator")
    c = _finite_matrix(observer, "observer")
    if a.shape[0] != a.shape[1]:
        raise ValueError("generator must be square")
    if c.shape[1] != a.shape[0]:
        raise ValueError("observer columns must match the state dimension")

    blocks = []
    propagated = c.copy()
    for _ in range(a.shape[0]):
        blocks.append(propagated)
        propagated = propagated @ a
    matrix = np.vstack(blocks)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if tolerance is None:
        scale = singular_values[0] if singular_values.size else 0.0
        tolerance = max(matrix.shape) * np.finfo(float).eps * scale
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    rank = int(np.sum(singular_values > tolerance))
    return LinearObservabilityCertificate(
        state_dimension=a.shape[0],
        observation_dimension=c.shape[0],
        rank=rank,
        tolerance=float(tolerance),
        singular_values=tuple(float(value) for value in singular_values),
        is_state_observable=rank == a.shape[0],
        matrix=matrix.copy(),
        scope=scope,
    )


def observer_ablation_ranks(
    generator,
    channels: Mapping[str, np.ndarray],
    *,
    tolerance: float | None = None,
) -> dict[str, int]:
    """Return full and one-channel-removed observability ranks.

    Every channel receives the same generator, precision and rank policy.  The
    result records ``all`` and ``without:<name>`` entries; it does not rank or
    optimize observers using the target response.
    """
    if not channels:
        raise ValueError("at least one observation channel is required")
    names = tuple(channels)
    matrices = {name: _finite_matrix(channels[name], name) for name in names}

    def rank_for(selected: tuple[str, ...]) -> int:
        if not selected:
            return 0
        observer = np.vstack([matrices[name] for name in selected])
        return linear_observability_certificate(
            generator, observer, tolerance=tolerance
        ).rank

    ranks = {"all": rank_for(names)}
    for omitted in names:
        selected = tuple(name for name in names if name != omitted)
        ranks[f"without:{omitted}"] = rank_for(selected)
    return ranks


def transform_linear_observer(
    generator, observer, basis
) -> tuple[np.ndarray, np.ndarray]:
    r"""Express ``(A,C)`` in an orthonormal state basis ``x = Q z``.

    This supports relabeling and degenerate-eigenspace rotation controls
    without selecting a privileged eigenvector inside a degenerate subspace.
    """
    a = _finite_matrix(generator, "generator")
    c = _finite_matrix(observer, "observer")
    q = _finite_matrix(basis, "basis")
    if a.shape[0] != a.shape[1] or q.shape != a.shape:
        raise ValueError("generator and basis must be square with equal shape")
    if c.shape[1] != a.shape[0]:
        raise ValueError("observer columns must match the state dimension")
    identity = np.eye(q.shape[0])
    tolerance = max(q.shape) * np.finfo(float).eps
    if not np.allclose(q.T @ q, identity, atol=tolerance, rtol=0.0):
        raise ValueError("basis must be orthonormal")
    return q.T @ a @ q, c @ q
