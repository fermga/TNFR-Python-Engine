r"""Exact level-set geometry of the constitutive coherence kernel.

For signed local coordinates p=DeltaNFR and v=dEPI, canonical coherence is

    C = 1 / (1 + |p| + |v|).

The local non-equilibrium levels are L1 diamonds.  For a fixed nonempty
network of N nodes, the canonical mean aggregation instead gives a
2N-dimensional cross-polytope.  A second certificate intersects that ambient
observation geometry with the nodal equation v_i = nu_f_i p_i for declared
fixed capacities.

All results in this module concern one instantaneous state.  Level-set
geometry alone proves neither temporal monotonicity nor attraction.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence, Set
from dataclasses import dataclass
import math
from numbers import Integral

from ._helpers import finite_real_scalar

__all__ = [
    "CoherenceLevelSetCertificate",
    "CrossPolytopeStratification",
    "FixedCapacityCoherenceLevelSetCertificate",
    "NetworkCoherenceLevelSetCertificate",
    "coherence_level_set_geometry",
    "fixed_capacity_coherence_level_set_geometry",
    "network_coherence_level_set_geometry",
]


_ERROR_COHERENCE = "coherence must be a finite scalar in (0, 1]"


@dataclass(frozen=True)
class CrossPolytopeStratification:
    """Exact face stratification of one L1-sphere level.

    A positive-radius L1 sphere in coordinate dimension d is the boundary of a
    d-dimensional cross-polytope.  Its k-dimensional faces number

        2**(k + 1) * binomial(d, k + 1).

    The relative interiors of its facets form the regular locus.  Lower faces
    form the nonsmooth locus when d is at least two.  A zero-radius level is a
    single smooth zero-dimensional set, although the absolute-value kernel is
    not differentiable at that point.
    """

    coordinate_dimension: int
    intrinsic_dimension: int
    vertex_count: int
    is_degenerate_point: bool
    singular_face_dimension_range: tuple[int, int] | None
    regular_facet_count_exponent: int | None
    level_set_is_globally_smooth_embedded_manifold: bool
    kernel_has_regular_locus: bool
    kernel_is_differentiable_everywhere_on_level: bool

    def face_count(self, face_dimension: int) -> int:
        """Return the exact number of faces of a requested dimension."""

        if isinstance(face_dimension, bool) or not isinstance(
            face_dimension, Integral
        ):
            raise ValueError("face_dimension must be an integer")
        dimension = int(face_dimension)
        if self.is_degenerate_point:
            if dimension != 0:
                raise ValueError(
                    "the degenerate level has only one zero-dimensional face"
                )
            return 1
        if not 0 <= dimension < self.coordinate_dimension:
            raise ValueError(
                "face_dimension must lie in [0, coordinate_dimension)"
            )
        return (1 << (dimension + 1)) * math.comb(
            self.coordinate_dimension, dimension + 1
        )

    def regular_facet_count(self) -> int:
        """Return the exact number of open regular facets."""

        if self.regular_facet_count_exponent is None:
            return 0
        return 1 << self.regular_facet_count_exponent


@dataclass(frozen=True)
class CoherenceLevelSetCertificate:
    """Geometry forced by one value of the local coherence kernel."""

    coherence: float
    l1_radius: float
    euclidean_radius_minimum: float
    euclidean_radius_maximum: float
    intrinsic_dimension: int
    singular_points: tuple[tuple[float, float], ...]
    smooth_away_from_singular_points: bool
    is_globally_smooth_embedded_manifold: bool
    is_equilibrium_point: bool
    regular_gradient_available: bool
    regular_gradient_norm: float
    claim_status: str


@dataclass(frozen=True)
class NetworkCoherenceLevelSetCertificate:
    """Fixed-node-count geometry of canonical network coherence.

    Coordinates are ordered as all pressure values followed by all observed
    EPI-rate values.  The certificate describes the complete ambient chart
    consumed by compute_coherence; it does not require those observations to
    satisfy the nodal equation.
    """

    coherence: float
    node_count: int
    ambient_dimension: int
    mean_l1_radius: float
    total_l1_radius: float
    euclidean_radius_minimum: float
    euclidean_radius_maximum: float
    stratification: CrossPolytopeStratification
    regular_gradient_available: bool
    regular_gradient_norm: float
    superlevel_is_closed_convex: bool
    temporal_attraction_certified: bool
    claim_status: str


@dataclass(frozen=True)
class FixedCapacityCoherenceLevelSetCertificate:
    """Network coherence level restricted by the fixed-capacity nodal law.

    With p_i=DeltaNFR_i and v_i=nu_f_i p_i, the level condition is

        sum_i (1 + nu_f_i) |p_i| = N * (1 / C - 1).

    It is therefore a weighted cross-polytope in pressure coordinates.  The
    Euclidean radii reported here use the metric induced by its embedding
    p_i -> (p_i, nu_f_i p_i) in the ambient pressure/rate chart.
    """

    coherence: float
    capacities: tuple[float, ...]
    node_count: int
    ambient_dimension: int
    nodal_slice_dimension: int
    mean_l1_radius: float
    weighted_pressure_radius: float
    pressure_axis_vertex_magnitudes: tuple[float, ...]
    embedded_axis_vertex_radii: tuple[float, ...]
    euclidean_radius_minimum: float
    euclidean_radius_maximum: float
    stratification: CrossPolytopeStratification
    regular_gradient_available: bool
    regular_gradient_norm: float
    nodal_equation_enforced: bool
    temporal_attraction_certified: bool
    claim_status: str


def _coherence_and_radius(coherence: float) -> tuple[float, float]:
    try:
        value = finite_real_scalar(coherence, "coherence")
    except ValueError as exc:
        raise ValueError(_ERROR_COHERENCE) from exc
    if not 0.0 < value <= 1.0:
        raise ValueError(_ERROR_COHERENCE)
    radius = (1.0 - value) / value
    if not math.isfinite(radius):
        raise ValueError("coherence level radius exceeds finite range")
    return value, radius


def _positive_node_count(node_count: int) -> int:
    if isinstance(node_count, bool) or not isinstance(node_count, Integral):
        raise ValueError("node_count must be a positive integer")
    normalized = int(node_count)
    if normalized <= 0:
        raise ValueError("node_count must be a positive integer")
    return normalized


def _cross_polytope_stratification(
    coordinate_dimension: int, *, equilibrium: bool
) -> CrossPolytopeStratification:
    if equilibrium:
        return CrossPolytopeStratification(
            coordinate_dimension=coordinate_dimension,
            intrinsic_dimension=0,
            vertex_count=1,
            is_degenerate_point=True,
            singular_face_dimension_range=None,
            regular_facet_count_exponent=None,
            level_set_is_globally_smooth_embedded_manifold=True,
            kernel_has_regular_locus=False,
            kernel_is_differentiable_everywhere_on_level=False,
        )

    singular_range = (
        (0, coordinate_dimension - 2)
        if coordinate_dimension >= 2
        else None
    )
    return CrossPolytopeStratification(
        coordinate_dimension=coordinate_dimension,
        intrinsic_dimension=coordinate_dimension - 1,
        vertex_count=2 * coordinate_dimension,
        is_degenerate_point=False,
        singular_face_dimension_range=singular_range,
        regular_facet_count_exponent=coordinate_dimension,
        level_set_is_globally_smooth_embedded_manifold=(
            coordinate_dimension == 1
        ),
        kernel_has_regular_locus=True,
        kernel_is_differentiable_everywhere_on_level=(
            coordinate_dimension == 1
        ),
    )


def _checked_regular_gradient_norm(value: float, scale: float) -> float:
    result = scale * value * value
    if result == 0.0 or not math.isfinite(result):
        raise ValueError(
            "regular gradient norm is outside finite floating-point range"
        )
    return result


def _checked_total_radius(radius: float, node_count: int) -> float:
    try:
        total = radius * node_count
    except OverflowError as exc:
        raise ValueError("network coherence level radius exceeds finite range") from exc
    if not math.isfinite(total):
        raise ValueError("network coherence level radius exceeds finite range")
    return total


def coherence_level_set_geometry(coherence: float) -> CoherenceLevelSetCertificate:
    r"""Return the exact local C=c level geometry.

    For 0<c<1, |DeltaNFR|+|dEPI|=1/c-1.  The Euclidean distance to equilibrium
    varies between r/sqrt(2) and r, while the L1 distance is exactly r.  At c=1
    the level set is the single equilibrium point.  The singleton is a smooth
    zero-dimensional embedded set, but the constitutive kernel has no gradient
    there.  No finite state attains c=0.
    """

    value, radius = _coherence_and_radius(coherence)
    if radius == 0.0:
        return CoherenceLevelSetCertificate(
            coherence=value,
            l1_radius=0.0,
            euclidean_radius_minimum=0.0,
            euclidean_radius_maximum=0.0,
            intrinsic_dimension=0,
            singular_points=((0.0, 0.0),),
            smooth_away_from_singular_points=False,
            is_globally_smooth_embedded_manifold=True,
            is_equilibrium_point=True,
            regular_gradient_available=False,
            regular_gradient_norm=0.0,
            claim_status=(
                "EXACT constitutive equilibrium level; temporal attraction "
                "is not certified"
            ),
        )

    regular_gradient_norm = _checked_regular_gradient_norm(
        value, math.sqrt(2.0)
    )
    singular = (
        (radius, 0.0),
        (-radius, 0.0),
        (0.0, radius),
        (0.0, -radius),
    )
    return CoherenceLevelSetCertificate(
        coherence=value,
        l1_radius=radius,
        euclidean_radius_minimum=radius / math.sqrt(2.0),
        euclidean_radius_maximum=radius,
        intrinsic_dimension=1,
        singular_points=singular,
        smooth_away_from_singular_points=True,
        is_globally_smooth_embedded_manifold=False,
        is_equilibrium_point=False,
        regular_gradient_available=True,
        regular_gradient_norm=regular_gradient_norm,
        claim_status=(
            "EXACT local ambient kernel geometry; network and fixed-capacity "
            "extensions are separate, and temporal attraction is not certified"
        ),
    )


def network_coherence_level_set_geometry(
    coherence: float, node_count: int
) -> NetworkCoherenceLevelSetCertificate:
    r"""Return the exact fixed-N ambient level geometry of network C(t).

    The network aggregator obeys

        C = 1 / (1 + (sum_i |p_i| + sum_i |v_i|) / N).

    Hence a non-equilibrium level is the boundary of a 2N-dimensional
    cross-polytope with total L1 radius N(1/C-1).  Its regular locus is the
    union of the relative interiors of 2**(2N) facets.  Every lower-dimensional
    face is a nonsmooth stratum of the ambient absolute-value kernel.
    """

    count = _positive_node_count(node_count)
    value, mean_radius = _coherence_and_radius(coherence)
    total_radius = _checked_total_radius(mean_radius, count)
    dimension = 2 * count
    equilibrium = total_radius == 0.0
    stratification = _cross_polytope_stratification(
        dimension, equilibrium=equilibrium
    )
    if equilibrium:
        minimum = maximum = gradient_norm = 0.0
        gradient_available = False
    else:
        minimum = total_radius / math.sqrt(float(dimension))
        if minimum == 0.0:
            raise ValueError(
                "network Euclidean radius is outside finite floating-point range"
            )
        maximum = total_radius
        gradient_norm = _checked_regular_gradient_norm(
            value, math.sqrt(2.0 / count)
        )
        gradient_available = True

    return NetworkCoherenceLevelSetCertificate(
        coherence=value,
        node_count=count,
        ambient_dimension=dimension,
        mean_l1_radius=mean_radius,
        total_l1_radius=total_radius,
        euclidean_radius_minimum=minimum,
        euclidean_radius_maximum=maximum,
        stratification=stratification,
        regular_gradient_available=gradient_available,
        regular_gradient_norm=gradient_norm,
        superlevel_is_closed_convex=True,
        temporal_attraction_certified=False,
        claim_status=(
            "EXACT fixed-node-count ambient C(t) geometry; nodal consistency, "
            "other state channels, changing support and temporal attraction "
            "are not implied"
        ),
    )


def fixed_capacity_coherence_level_set_geometry(
    coherence: float, capacities: Sequence[float]
) -> FixedCapacityCoherenceLevelSetCertificate:
    r"""Restrict a network coherence level to dEPI_i=nu_f_i*DeltaNFR_i.

    Capacities must be a concrete reusable nonempty sequence of finite
    nonnegative values. Mappings, sets and one-shot iterables are rejected.
    For fixed capacities, the nodal equation turns the ambient 2N-coordinate
    level into a weighted L1 sphere in N pressure coordinates. This is an
    instantaneous
    consistency slice, not an evolution or convergence certificate.
    """

    if (
        isinstance(
            capacities,
            (str, bytes, bytearray, Mapping, Set, Iterator),
        )
        or not isinstance(capacities, Sequence)
    ):
        raise ValueError("capacities must be a nonempty reusable sequence")
    raw_capacities = tuple(capacities)
    if not raw_capacities:
        raise ValueError("capacities must be a nonempty sequence")

    normalized: list[float] = []
    for index, capacity in enumerate(raw_capacities):
        try:
            capacity_value = finite_real_scalar(
                capacity, f"capacities[{index}]"
            )
        except ValueError as exc:
            raise ValueError(
                "capacities must contain finite nonnegative real scalars"
            ) from exc
        if capacity_value < 0.0:
            raise ValueError(
                "capacities must contain finite nonnegative real scalars"
            )
        normalized.append(capacity_value)

    nu_f = tuple(normalized)
    count = len(nu_f)
    value, mean_radius = _coherence_and_radius(coherence)
    weighted_radius = _checked_total_radius(mean_radius, count)
    equilibrium = weighted_radius == 0.0
    stratification = _cross_polytope_stratification(
        count, equilibrium=equilibrium
    )

    if equilibrium:
        pressure_vertices: tuple[float, ...] = ()
        embedded_vertices: tuple[float, ...] = ()
        minimum = maximum = gradient_norm = 0.0
        gradient_available = False
    else:
        pressure_values: list[float] = []
        embedded_values: list[float] = []
        dual_metric_terms: list[float] = []
        for capacity in nu_f:
            weight = 1.0 + capacity
            pressure_vertex = weighted_radius / weight
            if pressure_vertex == 0.0 or not math.isfinite(pressure_vertex):
                raise ValueError(
                    "nodal-slice pressure vertex is outside finite "
                    "floating-point range"
                )
            embedding_scale = math.hypot(1.0, capacity)
            embedded_radius = weighted_radius * (embedding_scale / weight)
            if not math.isfinite(embedded_radius):
                raise ValueError(
                    "nodal-slice Euclidean radius exceeds finite range"
                )
            pressure_values.append(pressure_vertex)
            embedded_values.append(embedded_radius)
            dual_metric_terms.append((weight / embedding_scale) ** 2)

        pressure_vertices = tuple(pressure_values)
        embedded_vertices = tuple(embedded_values)
        dual_norm_scale = math.sqrt(math.fsum(dual_metric_terms))
        minimum = weighted_radius / dual_norm_scale
        maximum = max(embedded_vertices)
        if minimum == 0.0 or not math.isfinite(minimum):
            raise ValueError(
                "nodal-slice Euclidean radius is outside finite "
                "floating-point range"
            )
        gradient_norm = _checked_regular_gradient_norm(
            value, dual_norm_scale / count
        )
        gradient_available = True

    return FixedCapacityCoherenceLevelSetCertificate(
        coherence=value,
        capacities=nu_f,
        node_count=count,
        ambient_dimension=2 * count,
        nodal_slice_dimension=count,
        mean_l1_radius=mean_radius,
        weighted_pressure_radius=weighted_radius,
        pressure_axis_vertex_magnitudes=pressure_vertices,
        embedded_axis_vertex_radii=embedded_vertices,
        euclidean_radius_minimum=minimum,
        euclidean_radius_maximum=maximum,
        stratification=stratification,
        regular_gradient_available=gradient_available,
        regular_gradient_norm=gradient_norm,
        nodal_equation_enforced=True,
        temporal_attraction_certified=False,
        claim_status=(
            "EXACT fixed-capacity nodal-equation slice of one instantaneous "
            "network C(t) level; capacity evolution, topology and attraction "
            "remain outside the certificate"
        ),
    )
