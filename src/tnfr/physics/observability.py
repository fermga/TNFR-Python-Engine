"""Scoped linear observability certificates for TNFR read-outs.

These helpers concern a declared finite-dimensional linear model ``x_dot=A x``
and observation ``y=C x``.  They do not assert that the canonical tetrad is a
complete state basis, nor do they linearize wrapped phase fields implicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ._helpers import finite_real_scalar

__all__ = [
    "EpiDiffusionReconstructionCertificate",
    "LinearObservabilityCertificate",
    "LocalObserverCertificate",
    "ObservationSignature",
    "finite_difference_observer_certificate",
    "epi_diffusion_reconstruction_certificate",
    "linear_observability_certificate",
    "minimal_distinguishing_channels",
    "observation_signature",
    "observer_ablation_ranks",
    "tetrad_observation_channels",
    "tetrad_observation_vector",
    "transform_linear_observer",
]


def _reject_boolean_numeric(value: Any, name: str) -> None:
    """Reject booleans before NumPy can coerce them to zero or one."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric, not boolean")


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


@dataclass(frozen=True)
class EpiDiffusionReconstructionCertificate:
    """Graph-specific reconstruction rank for the pure EPI channel.

    The full nodal structural-potential field observes ``-K L_rw x``.  The
    augmented observer adds the conserved ``d_i/nu_i`` mean. Exact-arithmetic
    full rank proves algebraic injectivity only for this fixed graph and
    frozen-capacity model. ``reconstructs_absolute_epi`` additionally requires
    scale-aware numerical rank and a small reconstruction residual.
    """

    nodes: tuple[Any, ...]
    state_dimension: int
    potential_rank: int
    potential_nullity: int
    augmented_rank: int
    relative_rank_tolerance: float
    potential_rank_tolerance: float
    augmented_rank_tolerance: float
    singular_values: tuple[float, ...]
    augmented_singular_values: tuple[float, ...]
    potential_condition_number: float
    potential_operator: np.ndarray
    mean_observer: np.ndarray
    reconstruction_residual: float
    relative_reconstruction_residual: float
    reconstruction_relative_tolerance: float
    reconstructs_modulo_uniform_shift: bool
    augmented_full_rank: bool
    reconstructs_absolute_epi: bool
    scope: str

    @property
    def tolerance(self) -> float:
        """Compatibility alias for the relative SVD rank tolerance."""
        return self.relative_rank_tolerance


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
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite two-dimensional matrix") from exc
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite two-dimensional matrix")
    return matrix


def _finite_l2_norm(value, name: str) -> float:
    """Return a scale-safe Euclidean norm or reject an unrepresentable value."""
    vector = np.asarray(value, dtype=float)
    scale = float(np.max(np.abs(vector), initial=0.0))
    if scale == 0.0:
        return 0.0
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            result = scale * float(np.linalg.norm(vector / scale))
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        raise ValueError(f"{name} exceeds finite floating-point range") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} exceeds finite floating-point range")
    return result


def _finite_observer_output(
    observer: Callable[[np.ndarray], np.ndarray],
    point: np.ndarray,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Evaluate an observer and snapshot a finite real one-dimensional output."""
    output = observer(np.array(point, dtype=float, copy=True))
    if np.iscomplexobj(output):
        raise ValueError("observer must return a finite real one-dimensional vector")
    try:
        value = np.asarray(output, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "observer must return a finite real one-dimensional vector"
        ) from exc
    if value.ndim != 1 or not np.all(np.isfinite(value)):
        raise ValueError("observer must return a finite real one-dimensional vector")
    if expected_shape is not None and value.shape != expected_shape:
        raise ValueError("observer output shape must remain fixed")
    # Snapshot mutable work buffers before another observer invocation.
    return np.array(value, dtype=float, copy=True)


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
            finite_real_scalar(
                get_attr(
                    graph.nodes[node],
                    ALIAS_VF,
                    0.0,
                    strict=True,
                    conv=lambda value: value,
                ),
                "structural frequency",
            )
            for node in nodes
        )
        if include_capacity else None
    )
    pressure = (
        tuple(
            finite_real_scalar(
                get_attr(
                    graph.nodes[node],
                    ALIAS_DNFR,
                    0.0,
                    strict=True,
                    conv=lambda value: value,
                ),
                "DeltaNFR",
            )
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


def epi_diffusion_reconstruction_certificate(
    graph: Any,
    *,
    tolerance: float | None = None,
) -> EpiDiffusionReconstructionCertificate:
    r"""Test whether full ``Phi_s`` plus one zero-mode scalar reconstructs EPI.

    For fixed connected undirected conductance, pure EPI pressure is
    ``p=-L_rw x``.  The exact inverse-square structural-potential field is
    ``Phi_s=Kp``, so its linear observation operator is ``O=-K L_rw``.
    Constants always belong to ``ker(O)``.  Rank ``N-1`` means that this is the
    only invisible EPI direction.  Positive frozen capacities supply the
    conserved mean row proportional to ``d_i/nu_i``; appending it makes the
    observer injective in exact arithmetic. The published
    ``reconstructs_absolute_epi`` decision additionally requires scale-aware
    numerical rank and the declared reconstruction-residual bound.

    ``tolerance`` is a dimensionless relative SVD cutoff and is applied
    separately to the potential and augmented observers.  This separation
    prevents a rescaling of the potential block from reusing an incompatible
    absolute threshold after the conserved-mean row is appended.
    The conserved weights are max-scaled before normalization.  If a positive
    observer coefficient or inverse-square kernel entry cannot be represented,
    the routine rejects the certificate instead of silently replacing it by
    zero or infinity.

    The certificate uses the same weighted shortest-path convention as
    ``compute_structural_potential``.  It is graph-specific and does not claim
    that every graph has rank ``N-1``.  Capacity remains a required model
    parameter for predicting the generator, even when EPI is reconstructible.
    """
    import networkx as nx

    from ..alias import get_attr
    from ..constants.aliases import ALIAS_VF
    from ..mathematics._weight_normalization import normalize_weights
    from ._conductance import read_conductance
    from ._edge_semantics import structural_path_weight
    from .structural_diffusion import structural_diffusion_operator, structural_field

    if tolerance is not None:
        _reject_boolean_numeric(tolerance, "tolerance")
    if graph.is_directed():
        raise ValueError("EPI reconstruction certificate requires an undirected graph")
    conductance = read_conductance(graph, symmetric=True)
    nodes = tuple(conductance.nodes)
    n = len(nodes)
    if n < 2:
        raise ValueError("EPI reconstruction certificate requires at least two nodes")
    strength = conductance.strength
    if np.any(strength <= 0.0):
        raise ValueError(
            "EPI reconstruction certificate requires positive row strength"
        )
    effective_graph = nx.Graph()
    effective_graph.add_nodes_from(range(n))
    effective_graph.add_edges_from(
        (int(source), int(target))
        for source, target in zip(conductance.source, conductance.target)
        if source != target
    )
    if not nx.is_connected(effective_graph):
        raise ValueError(
            "EPI reconstruction certificate requires connected positive conductance"
        )

    frequency = np.array(
        [
            finite_real_scalar(
                get_attr(
                    graph.nodes[node],
                    ALIAS_VF,
                    0.0,
                    conv=lambda value: value,
                    strict=True,
                ),
                "structural frequency",
            )
            for node in nodes
        ],
        dtype=float,
    )
    if np.any(frequency <= 0.0):
        raise ValueError(
            "EPI reconstruction certificate requires positive finite capacity"
        )
    field = structural_field(graph, list(nodes))
    if not np.all(np.isfinite(field)):
        raise ValueError("EPI reconstruction certificate requires finite scalar EPI")

    try:
        distances = dict(
            nx.all_pairs_dijkstra_path_length(
                graph, weight=structural_path_weight(graph)
            )
        )
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(
            "Structural-potential distances exceed finite floating-point range"
        ) from exc
    kernel = np.zeros((n, n), dtype=float)
    for source_index, source in enumerate(nodes):
        for target_index, target in enumerate(nodes):
            distance = distances[source].get(target)
            if (source_index != target_index and distance is not None
                    and np.isfinite(distance) and distance > 0.0):
                try:
                    with np.errstate(over="raise", invalid="raise", divide="raise"):
                        inverse_distance = 1.0 / float(distance)
                        kernel_value = inverse_distance * inverse_distance
                except (FloatingPointError, OverflowError, ZeroDivisionError) as exc:
                    raise ValueError(
                        "Structural-potential kernel exceeds finite "
                        "floating-point range"
                    ) from exc
                if not np.isfinite(kernel_value) or kernel_value <= 0.0:
                    raise ValueError(
                        "Structural-potential kernel exceeds finite "
                        "floating-point range"
                    )
                kernel[source_index, target_index] = kernel_value
    if not np.all(np.isfinite(kernel)):
        raise ValueError("Structural-potential kernel exceeds floating-point range")

    _, laplacian = structural_diffusion_operator(graph)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            potential_operator = -(kernel @ laplacian)
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(
            "Structural-potential observer exceeds finite floating-point range"
        ) from exc
    if not np.all(np.isfinite(potential_operator)):
        raise ValueError(
            "Structural-potential observer exceeds finite floating-point range"
        )
    if tolerance is None:
        tolerance = max(potential_operator.shape) * np.finfo(float).eps
    if not np.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
        raise ValueError("tolerance must be finite and in the open interval (0, 1)")
    try:
        singular_values = np.linalg.svd(potential_operator, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Structural-potential observer SVD did not converge") from exc
    if not np.all(np.isfinite(singular_values)):
        raise ValueError("Structural-potential observer SVD is non-finite")
    potential_rank_tolerance = (
        float(tolerance * singular_values[0]) if singular_values.size else 0.0
    )
    potential_rank = int(np.sum(singular_values > potential_rank_tolerance))

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            metric_weights = strength / frequency
    except (FloatingPointError, OverflowError) as exc:
        raise ValueError(
            "Reconstruction metric weights exceed finite floating-point range"
        ) from exc
    if (not np.all(np.isfinite(metric_weights))
            or np.any(metric_weights <= 0.0)):
        raise ValueError("Reconstruction metric weights must be finite and positive")
    mean_observer, _, _ = normalize_weights(metric_weights)
    if not np.all(np.isfinite(mean_observer)) or np.any(mean_observer <= 0.0):
        raise ValueError(
            "Reconstruction metric normalization exceeds floating-point dynamic range"
        )
    mean_observer = mean_observer[None, :]
    augmented = np.vstack((potential_operator, mean_observer))
    try:
        augmented_singular_values = np.linalg.svd(augmented, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Augmented reconstruction observer SVD did not converge"
        ) from exc
    if not np.all(np.isfinite(augmented_singular_values)):
        raise ValueError("Augmented reconstruction observer SVD is non-finite")
    augmented_rank_tolerance = (
        float(tolerance * augmented_singular_values[0])
        if augmented_singular_values.size else 0.0
    )
    augmented_rank = int(
        np.sum(augmented_singular_values > augmented_rank_tolerance)
    )

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            observation = np.concatenate(
                (potential_operator @ field, mean_observer @ field)
            )
            reconstructed = (
                np.linalg.pinv(augmented, rcond=float(tolerance)) @ observation
            )
            reconstruction_error = reconstructed - field
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        raise ValueError(
            "EPI reconstruction exceeds finite floating-point range"
        ) from exc
    if (not np.all(np.isfinite(observation))
            or not np.all(np.isfinite(reconstructed))
            or not np.all(np.isfinite(reconstruction_error))):
        raise ValueError("EPI reconstruction exceeds finite floating-point range")
    residual = _finite_l2_norm(reconstruction_error, "EPI reconstruction residual")
    field_norm = _finite_l2_norm(field, "EPI field norm")
    relative_residual = residual / max(1.0, field_norm)
    modulo_shift = potential_rank == n - 1
    augmented_full_rank = augmented_rank == n
    reconstruction_relative_tolerance = math.sqrt(float(tolerance))
    absolute = (
        augmented_full_rank
        and np.isfinite(relative_residual)
        and relative_residual <= reconstruction_relative_tolerance
    )
    condition_number = (
        float(singular_values[0] / singular_values[n - 2])
        if modulo_shift and singular_values[n - 2] > 0.0
        else float("inf")
    )
    return EpiDiffusionReconstructionCertificate(
        nodes=nodes,
        state_dimension=n,
        potential_rank=potential_rank,
        potential_nullity=n - potential_rank,
        augmented_rank=augmented_rank,
        relative_rank_tolerance=float(tolerance),
        potential_rank_tolerance=potential_rank_tolerance,
        augmented_rank_tolerance=augmented_rank_tolerance,
        singular_values=tuple(float(value) for value in singular_values),
        augmented_singular_values=tuple(
            float(value) for value in augmented_singular_values
        ),
        potential_condition_number=condition_number,
        potential_operator=potential_operator.copy(),
        mean_observer=mean_observer.copy(),
        reconstruction_residual=residual,
        relative_reconstruction_residual=relative_residual,
        reconstruction_relative_tolerance=reconstruction_relative_tolerance,
        reconstructs_modulo_uniform_shift=modulo_shift,
        augmented_full_rank=augmented_full_rank,
        reconstructs_absolute_epi=absolute,
        scope="fixed connected undirected pure EPI diffusion",
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
    _reject_boolean_numeric(step, "step")
    if tolerance is not None:
        _reject_boolean_numeric(tolerance, "tolerance")
    point = np.asarray(state, dtype=float)
    if point.ndim != 1 or not np.all(np.isfinite(point)):
        raise ValueError("state must be a finite one-dimensional vector")
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("step must be finite and positive")
    baseline = _finite_observer_output(observer, point)
    jacobian = np.empty((baseline.size, point.size), dtype=float)
    for column in range(point.size):
        delta = np.zeros_like(point)
        delta[column] = step
        plus = _finite_observer_output(
            observer, point + delta, expected_shape=baseline.shape
        )
        minus = _finite_observer_output(
            observer, point - delta, expected_shape=baseline.shape
        )
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                derivative = (plus - minus) / (2.0 * step)
        except (FloatingPointError, OverflowError) as exc:
            raise ValueError(
                "finite-difference Jacobian exceeds floating-point range"
            ) from exc
        if not np.all(np.isfinite(derivative)):
            raise ValueError(
                "finite-difference Jacobian exceeds floating-point range"
            )
        jacobian[:, column] = derivative
    try:
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("finite-difference observer SVD did not converge") from exc
    if not np.all(np.isfinite(singular_values)):
        raise ValueError("finite-difference observer SVD is non-finite")
    if tolerance is None:
        scale = singular_values[0] if singular_values.size else 0.0
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                tolerance = (
                    max(jacobian.shape) * np.finfo(float).eps * scale / step
                )
        except (FloatingPointError, OverflowError) as exc:
            raise ValueError(
                "finite-difference rank tolerance exceeds floating-point range"
            ) from exc
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
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
    if tolerance is not None:
        _reject_boolean_numeric(tolerance, "tolerance")
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
    if tolerance is not None:
        _reject_boolean_numeric(tolerance, "tolerance")
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
