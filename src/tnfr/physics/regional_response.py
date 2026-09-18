"""Exact directional response of a region under a declared affine difference map.

The full positive metric and transition are inputs, not fitted from endpoints.
For paired nodal trajectories an adapter must first identify a common map and
account for source differences and realization defects. This detached observer
does not execute or authenticate an operator, predict a policy, or identify NFRs.
"""

from collections.abc import Mapping, Set
from dataclasses import dataclass
from fractions import Fraction as F

from ..mathematics.krylov import exact_rank
from ._cycle_algebra import Matrix, Vector, dot, ordered_vector
from ._exact_linear_algebra import exact_matrix_inverse, exact_square_matrix_product

__all__ = [
    "RegionalResponseCriterion", "observe_regional_response",
    "RegionalInputGeometry", "observe_regional_input_geometry",
]


@dataclass(frozen=True)
class RegionalResponseCriterion:
    """Conditional map identity and signed input criterion in one fixed metric.

    ``norm_only_sufficient`` certifies the input ball using its displayed norm;
    it is not necessary for the particular input direction. A measured residual
    makes this a retrospective check, not an independent future-error bound.
    ``nullspace_preserved`` is necessary for a finite all-state regional gain,
    but does not by itself imply nonincrease or contraction.
    """

    transition: Matrix
    metric_weights: Vector
    region_indices: tuple[int, ...]
    difference: Vector
    residual: Vector
    centering: Matrix
    energy_matrix: Matrix
    change_matrix: Matrix
    child_mean: F
    parent_mean: F
    centered_before: Vector
    centered_ideal: Vector
    centered_after: Vector
    self_image: Vector
    input_image: Vector
    input_components: tuple
    before_squared_norm: F
    self_squared_norm: F
    input_squared_norm: F
    input_signed_work: F
    available_squared_norm_drop: F
    ideal_energy_change: F
    residual_linear_energy: F
    residual_quadratic_energy: F
    energy_change: F
    nonincreasing: bool
    strict_decrease: bool
    norm_only_sufficient: bool
    nullspace_images: tuple
    nullspace_preserved: bool


def _regional_geometry(transition, metric_weights, region_indices):
    """One validated domain, centering and environmental basis for both owners."""
    metric = ordered_vector(metric_weights, "metric weights")
    size = len(metric)
    if size < 2 or any(h <= 0 for h in metric):
        raise ValueError("a positive full metric with at least two nodes is required")
    if isinstance(transition, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError("transition must be an ordered matrix")
    matrix = tuple(ordered_vector(row, "transition row") for row in transition)
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise ValueError("transition must be square and match the full metric")
    if isinstance(region_indices, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError("region indices must be an ordered sequence")
    region = tuple(region_indices)
    if (not region or len(region) >= size
            or any(type(i) is not int or not 0 <= i < size for i in region)
            or len(set(region)) != len(region)):
        raise ValueError("region must be a nonempty proper set of distinct indices")
    outside = tuple(i for i in range(size) if i not in region)
    mass = sum((metric[i] for i in region), F(0))
    # C is rectangular: restriction followed by regional weighted centering.
    center = tuple(tuple(F(i == j)-(metric[j]/mass if j in region else F(0))
                         for j in range(size)) for i in region)
    basis = (("region_constant", tuple(F(i in region) for i in range(size))),)
    basis += tuple((f"outside_{j}", tuple(F(i == j) for i in range(size))) for j in outside)
    return matrix, metric, region, outside, center, basis


def observe_regional_response(
    transition, metric_weights, region_indices, difference, *, residual=None,
):
    """Derive the response of ``delta_after = T delta_before + residual``.

    Restrict and center on a nonempty proper region B with the supplied positive
    full H metric. With ``delta=z_B+w`` and ``Cw=0``, set ``v=CTz_B`` and
    ``u=CTw+Cr``. Then nonincrease is equivalent to
    ``2<v,u>_H + ||u||_H^2 <= ||C delta||_H^2 - ||v||_H^2``.

    ``T`` can in particular be an admitted ``S-hA`` for a reset after pressure
    generation. Timing, common coefficients, source cancellation and the meaning
    of the residual belong to the caller's evidence, not to this algebra.
    No pressure is reconstructed from an observed derivative.
    """
    matrix, metric, region, outside, center, basis = _regional_geometry(
        transition, metric_weights, region_indices)
    size = len(metric)
    x = ordered_vector(difference, "difference")
    r = (F(0),)*size if residual is None else ordered_vector(residual, "residual")
    if len(x) != size or len(r) != size:
        raise ValueError("difference and residual must match the full metric")

    def mv(m, value):
        return tuple(dot(row, value) for row in m)

    def inner(a, b):
        return sum((metric[i]*u*v for i, u, v in zip(region, a, b, strict=True)), F(0))

    def mean(indices):
        return sum((metric[i]*x[i] for i in indices), F(0))/sum(metric[i] for i in indices)

    mb, mp = mean(region), mean(outside)
    z = tuple(x[i]-mb if i in region else F(0) for i in range(size))
    modes = (
        ("global_mean", (mp,)*size),
        ("mean_contrast", tuple(mb-mp if i in region else F(0) for i in range(size))),
        ("parent_centered", tuple(x[i]-mp if i in outside else F(0) for i in range(size))),
    )
    components = tuple((label, mv(center, mv(matrix, value))) for label, value in modes)
    components += (("runtime_residual", mv(center, r)),)
    before, ideal = mv(center, x), mv(center, mv(matrix, x))
    residual_centered = components[-1][1]
    after = tuple(a+b for a, b in zip(ideal, residual_centered, strict=True))
    self_image = mv(center, mv(matrix, z))
    incoming = tuple(sum((value[k] for _, value in components), F(0)) for k in range(len(region)))
    if tuple(a+b for a, b in zip(self_image, incoming, strict=True)) != after:
        raise RuntimeError("regional mean/shape/input identity failed")
    energy_matrix = tuple(tuple(sum((metric[i]*row[j]*row[k]
                                    for i, row in zip(region, center, strict=True)), F(0))
                                for k in range(size)) for j in range(size))
    transformed = exact_square_matrix_product(
        tuple(zip(*matrix, strict=True)), exact_square_matrix_product(energy_matrix, matrix))
    change = tuple(tuple(a-b for a, b in zip(row, old, strict=True))
                   for row, old in zip(transformed, energy_matrix, strict=True))
    zn, an, un = inner(before, before), inner(self_image, self_image), inner(incoming, incoming)
    work, available = 2*inner(self_image, incoming)+un, zn-an
    ideal_change = dot(x, mv(change, x))/2
    linear, quadratic = inner(ideal, residual_centered), inner(residual_centered, residual_centered)/2
    total = (inner(after, after)-zn)/2
    if ideal_change+linear+quadratic != total or (work-available)/2 != total:
        raise RuntimeError("regional quadratic response identity failed")
    # {1_B, e_j: j outside B} spans ker(C); no coordinate search is involved.
    images = tuple((label, mv(center, mv(matrix, value))) for label, value in basis)
    sufficient = zn >= an+un and (zn-an-un)**2 >= 4*an*un
    return RegionalResponseCriterion(
        matrix, metric, region, x, r, center, energy_matrix, change, mb, mp,
        before, ideal, after, self_image, incoming, components, zn, an, un,
        work, available, ideal_change, linear, quadratic, total,
        work <= available, work < available, sufficient, images,
        not any(any(value) for _, value in images),
    )


@dataclass(frozen=True)
class RegionalInputGeometry:
    """Exact image and protected read-outs for one declared transition.

    Protection is independence from arbitrary inputs in ker(C), not invariance
    under self-dynamics, realizability of those inputs, or protection from
    execution residuals. Image/basis choices follow the supplied coordinate
    order; the subspaces do not depend on that choice. No rank tolerance is used.
    """

    transition: Matrix
    metric_weights: Vector
    region_indices: tuple[int, ...]
    centering: Matrix
    input_labels: tuple[str, ...]
    input_basis: Matrix
    input_map: Matrix
    shape_dimension: int
    rank: int
    protected_dimension: int
    image_basis_indices: tuple[int, ...]
    image_basis: Matrix
    gram_inverse: Matrix
    input_coefficients: Matrix
    image_projection: Matrix
    protected_projection: Matrix
    protected_basis: Matrix
    protected_readout_rows: Matrix


def observe_regional_input_geometry(transition, metric_weights, region_indices):
    """Classify G=CTN and its H-orthogonal annihilator in centered shape space.

    N contains the regional indicator and outside coordinate vectors, spanning
    ker(C). For independent image columns V, the H-orthogonal image projector
    is V(V^T H_B V)^-1 V^T H_B. Subtracting it from the regional centering
    projector gives all protected centered read-outs. Shared exact rank and
    inverse primitives supply witnesses without spectral tolerances or fitting.
    """
    matrix, metric, region, _outside, center, basis = _regional_geometry(
        transition, metric_weights, region_indices)
    count = len(region)
    weights = tuple(metric[i] for i in region)

    def mv(m, value):
        return tuple(dot(row, value) for row in m)

    def inner(a, b):
        return sum((h*x*y for h, x, y in zip(weights, a, b, strict=True)), F(0))

    def independent(vectors):
        chosen, indices = [], []
        for index, value in enumerate(vectors):
            rank = exact_rank((*chosen, value))
            if rank > len(chosen):
                chosen.append(value)
                indices.append(index)
        return tuple(indices), tuple(chosen)

    images = tuple(mv(center, mv(matrix, vector)) for _, vector in basis)
    if any(dot(weights, column) for column in images):
        raise RuntimeError("environmental input image is not H-centered")
    selected, image_basis = independent(images)
    rank = len(selected)
    gram = tuple(tuple(inner(a, b) for b in image_basis) for a in image_basis)
    inverse = exact_matrix_inverse(gram) if rank else ()
    coefficients_by_column = tuple(mv(inverse, tuple(inner(a, column) for a in image_basis))
                                   for column in images)
    coefficients = tuple(tuple(column[i] for column in coefficients_by_column) for i in range(rank))
    for column, coords in zip(images, coefficients_by_column, strict=True):
        rebuilt = tuple(sum((v[i]*a for v, a in zip(image_basis, coords, strict=True)), F(0))
                        for i in range(count))
        if rebuilt != column:
            raise RuntimeError("independent image columns do not reconstruct the input map")
    # At rank zero all sums are empty, giving the exact zero projection.
    projection = tuple(tuple(sum((image_basis[a][i]*inverse[a][b]*image_basis[b][j]*weights[j]
                                  for a in range(rank) for b in range(rank)), F(0))
                             for j in range(count)) for i in range(count))
    mass = sum(weights, F(0))
    local_center = tuple(tuple(F(i == j)-weights[j]/mass for j in range(count)) for i in range(count))
    protected = tuple(tuple(c-p for c, p in zip(cr, pr, strict=True))
                      for cr, pr in zip(local_center, projection, strict=True))
    _, protected_basis = independent(tuple(zip(*protected, strict=True)))
    dimension = count-1-rank
    if len(protected_basis) != dimension:
        raise RuntimeError("image and protected dimensions do not exhaust centered shape")
    readouts = tuple(tuple(h*x for h, x in zip(weights, value, strict=True)) for value in protected_basis)
    for value, readout in zip(protected_basis, readouts, strict=True):
        if dot(weights, value) or any(dot(readout, column) for column in images):
            raise RuntimeError("protected read-out fails its centered annihilator identity")
    for p in (projection, protected):
        if exact_square_matrix_product(p, p) != p or any(
                weights[i]*p[i][j] != weights[j]*p[j][i]
                for i in range(count) for j in range(count)):
            raise RuntimeError("regional projection is not H-orthogonal and idempotent")
    return RegionalInputGeometry(
        matrix, metric, region, center, tuple(label for label, _ in basis),
        tuple(value for _, value in basis), tuple(zip(*images, strict=True)),
        count-1, rank, dimension, selected, image_basis, inverse, coefficients,
        projection, protected, protected_basis, readouts,
    )
