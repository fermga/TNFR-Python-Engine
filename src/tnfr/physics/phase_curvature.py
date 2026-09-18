"""Read-only circular curvature from explicitly represented phasor components.

NumPy binary64 sine/cosine values are materialized once per primitive phase.
Their sums are exact rationals; the resulting angle and wrapped curvature are
ordinary floating approximations. This is neither exact trigonometry nor a
guarantee of well-conditioned direction near cancellation. In particular an
exactly zero represented sum need not be an exact-real phasor cancellation.
This kernel does not mutate state, evolve phase, or replace pressure/IL kernels.
Existing telemetry consumers can propagate its explicit domain failure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Any

from .._exact_time import finite_represented_real
from ..mathematics.phasor_resultant import (
    RepresentedPhasorResultant,
    reduce_phasor_components,
)
from ..mathematics.unified_numerical import np
from ._helpers import wrap_angle

__all__ = (
    "PhaseCurvatureNodeObservation",
    "PhaseCurvatureObservation",
    "UndefinedPhaseCurvatureError",
)


class UndefinedPhaseCurvatureError(ValueError):
    """A nonempty neighborhood has zero represented phasor resultant."""

    def __init__(self, nodes: tuple[Any, ...]):
        self.nodes = nodes
        super().__init__(
            f"Phase curvature is undefined at nodes {nodes!r}: exact represented "
            "joint-zero neighbor resultant; use observe_phase_curvature for evidence"
        )


@dataclass(frozen=True)
class PhaseCurvatureNodeObservation:
    """One unique-neighbor read-out; an isolate uses the explicit zero convention."""

    node: Any
    neighbors: tuple[Any, ...]
    gradient: float
    curvature: float | None
    status: str
    resultant: RepresentedPhasorResultant | None


@dataclass(frozen=True)
class PhaseCurvatureObservation:
    """Detached numerical evidence, with graph node identifiers retained shallowly.

    ``requested_precision_mode`` does not upgrade the binary64 trigonometry.
    ``gradient_accumulator_dtype`` records the separate neighborhood reduction.
    Component sums are permutation invariant for a fixed materialized multiset;
    rounding phases or their trigonometric values after a rotation is a distinct
    input and need not preserve a poorly conditioned direction.
    """

    version: str
    nodes: tuple[Any, ...]
    neighbor_order: tuple[tuple[Any, ...], ...]
    primitive_phases: tuple[float, ...]
    components: tuple[tuple[float, float], ...]
    requested_precision_mode: str
    gradient_accumulator_dtype: str
    rows: tuple[PhaseCurvatureNodeObservation, ...]
    scope: tuple[str, ...] = (
        "numpy_binary64_trigonometric_components",
        "exact_reduction_of_materialized_components_only",
        "approximate_atan2_and_wrapped_curvature",
        "nonzero_resultant_is_not_a_conditioning_certificate",
        "represented_zero_does_not_certify_exact_real_trigonometric_zero",
        "unweighted_unique_neighbors_including_zero_weight_edges",
        "readout_kernel_does_not_mutate_or_evolve_state",
    )


def _materialize_phases(values):
    return tuple(
        finite_represented_real(value, f"phase[{index}]")[0]
        for index, value in enumerate(values)
    )


def _observe_neighborhoods(nodes, neighbors, phases, *, dtype, precision_mode):
    """Shared graph/array kernel on already validated ordered neighborhoods."""
    phases = _materialize_phases(phases)
    theta = np.asarray(phases, dtype=np.float64)
    cosines, sines = np.cos(theta), np.sin(theta)
    components = tuple((float(c), float(s)) for c, s in zip(cosines, sines))
    rows = []
    neighbor_order = tuple(tuple(nodes[j] for j in indices) for indices in neighbors)
    for i, indices in enumerate(neighbors):
        if not indices:
            rows.append(
                PhaseCurvatureNodeObservation(
                    nodes[i],
                    (),
                    0.0,
                    0.0,
                    "isolated_zero_convention",
                    None,
                )
            )
            continue
        # Preserve wrapped angular separation, independently of whether a
        # circular mean exists. A finite subtraction outside binary64's
        # range is rejected rather than silently producing NaN telemetry.
        with np.errstate(over="ignore", invalid="ignore"):
            differences = theta[i] - theta[list(indices)]
        if not np.all(np.isfinite(differences)):
            raise ValueError("phase differences must be finite binary64 values")
        wrapped = (differences + np.pi) % (2 * np.pi) - np.pi
        gradient = float(np.mean(np.abs(wrapped), dtype=dtype))
        resultant = reduce_phasor_components(components[j] for j in indices)
        if resultant.joint_zero:
            curvature = None
            status = "undefined_represented_resultant"
        else:
            displacement = phases[i] - resultant.angle
            if not math.isfinite(displacement):
                raise ValueError("phase curvature displacement must be finite")
            curvature = float(wrap_angle(displacement))
            status = "defined"
        rows.append(
            PhaseCurvatureNodeObservation(
                nodes[i],
                neighbor_order[i],
                gradient,
                curvature,
                status,
                resultant,
            )
        )
    return PhaseCurvatureObservation(
        version="phase_curvature_exact_components_v1",
        nodes=tuple(nodes),
        neighbor_order=neighbor_order,
        primitive_phases=phases,
        components=components,
        requested_precision_mode=precision_mode,
        gradient_accumulator_dtype=np.dtype(dtype).name,
        rows=tuple(rows),
    )


def _require_defined_curvature(observation):
    undefined = tuple(row.node for row in observation.rows if row.curvature is None)
    if undefined:
        raise UndefinedPhaseCurvatureError(undefined)


def _observe_phase_arrays(
    theta_arr, edge_src, edge_dst, degrees, *, dtype, precision_mode
):
    """Validate the public array adapter's unique-neighbor incidence contract."""
    if np.ndim(theta_arr) != 1 or np.ndim(degrees) != 1:
        raise ValueError("phases and neighbor counts must be one-dimensional")
    n = len(theta_arr)
    if len(degrees) != n or np.ndim(edge_src) != 1 or np.ndim(edge_dst) != 1:
        raise ValueError("phase incidence arrays have incompatible shapes")
    if len(edge_src) != len(edge_dst):
        raise ValueError("phase source and destination arrays must align")
    neighbors = [[] for _ in range(n)]
    seen = set()
    for source, target in zip(edge_src, edge_dst):
        if any(
            isinstance(i, (bool, np.bool_))
            or not isinstance(i, Integral)
            or not 0 <= i < n
            for i in (source, target)
        ):
            raise ValueError("phase incidence indices must be in-range integers")
        pair = (int(target), int(source))
        if pair in seen:
            raise ValueError("phase incidence must contain unique neighbors")
        seen.add(pair)
        neighbors[int(target)].append(int(source))
    for index, value in enumerate(degrees):
        count = finite_represented_real(value, f"neighbor count[{index}]")[0]
        if count != len(neighbors[index]):
            raise ValueError("neighbor counts must match phase incidence")
    return _observe_neighborhoods(
        tuple(range(n)),
        tuple(tuple(row) for row in neighbors),
        theta_arr,
        dtype=dtype,
        precision_mode=precision_mode,
    )
