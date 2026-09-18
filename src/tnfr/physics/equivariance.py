r"""Equivariance certificates for the canonical emergent operator (R1).

Certifies the **diffusion-sector observability** result: the structural-diffusion
operator ``L_rw`` commutes with every automorphism of the graph (equivariance
residual ``max_σ ‖P_σ L − L P_σ‖ ≈ 0``) and preserves the Reynolds sectors
(``‖L Q_Γ − Q_Γ L‖ ≈ 0``).  Consequently ``Fix(Γ)`` and ``Fix(Γ)^⊥`` are
invariant under the overdamped nodal-equation flow ``ẋ = −D_νf · L_rw · x``
whenever ``νf`` is orbit-constant, so a symmetric seed evolved by the canonical
dynamics can never manufacture per-node structure that distinguishes nodes within
one orbit.  This is the proved, implementation-certified base case that the
operator-by-operator equivariance audit (a later R1 stage) builds on.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Set
from fractions import Fraction as F

import numpy as np

from .spectral_projectors import derived_tolerance
from ._cycle_algebra import Matrix, Vector, ordered_vector
from .structural_diffusion import structural_diffusion_operator
from .support_transport import SupportTransportSnapshot, _from_data
from .symmetry_sectors import (
    automorphism_orbits,
    automorphism_permutations,
    permutation_matrix,
    reynolds_projector,
)

__all__ = [
    "EquivarianceCertificate",
    "equivariance_residual",
    "sector_preservation_residual",
    "verify_diffusion_equivariance",
    "ExactSymmetryDefect", "ExactPermutationSymmetry", "ExactMapSymmetry",
    "observe_exact_map_symmetries",
]


def equivariance_residual(operator_matrix, permutations, nodes) -> float:
    r"""``max_σ ‖P_σ L − L P_σ‖₂`` over the given automorphism permutations."""
    L = np.asarray(operator_matrix)
    best = 0.0
    for mapping in permutations:
        P = permutation_matrix(mapping, nodes)
        best = max(best, float(np.linalg.norm(P @ L - L @ P, 2)))
    return best


def sector_preservation_residual(operator_matrix, projector) -> float:
    r"""``‖L Q_Γ − Q_Γ L‖₂``; zero iff ``L`` preserves ``Fix(Γ) ⊕ Fix(Γ)^⊥``."""
    L = np.asarray(operator_matrix)
    Q = np.asarray(projector)
    return float(np.linalg.norm(L @ Q - Q @ L, 2))


@dataclass(frozen=True)
class EquivarianceCertificate:
    r"""Certificate that ``L_rw`` is equivariant and sector-preserving on ``G``."""

    equivariance_residual: float
    sector_preservation_residual: float
    tolerance: float
    n_automorphisms: int
    orbit_count: int
    is_equivariant: bool


def verify_diffusion_equivariance(
    G,
    *,
    tol: float | None = None,
    weight: str | None = None,
    cap: int = 2000,
) -> EquivarianceCertificate:
    r"""Certify that ``L_rw`` commutes with ``Aut(G)`` and preserves its sectors.

    The tolerance defaults to the derived ``√ε·‖L‖₂``; the certificate reports
    both residuals, the number of automorphisms found and the orbit count.
    """
    nodes, L = structural_diffusion_operator(G)
    perms = automorphism_permutations(G, weight=weight, cap=cap)
    if tol is None:
        tol = derived_tolerance(L)
    eq = equivariance_residual(L, perms, nodes)
    Q = reynolds_projector(G, nodes=nodes, permutations=perms)
    pres = sector_preservation_residual(L, Q)
    orbits = automorphism_orbits(G, nodes=nodes, permutations=perms)
    return EquivarianceCertificate(
        equivariance_residual=eq,
        sector_preservation_residual=pres,
        tolerance=tol,
        n_automorphisms=len(perms),
        orbit_count=len(orbits),
        is_equivariant=(eq < tol and pres < tol),
    )


@dataclass(frozen=True)
class ExactSymmetryDefect:
    """Exact conjugation/field defect; witness is its first nonzero entry.

    Matrix entries compare M[p(i),p(j)]-M[i,j]; fields compare v[p(i)]-v[i].
    Zero conjugation defect is equivalent to permutation commutation. The
    maximum is an entrywise absolute defect, not a spectral norm.
    """

    preserved: bool
    max_abs_defect: F
    witness: tuple | None


@dataclass(frozen=True)
class ExactPermutationSymmetry:
    """One weighted-support permutation, with independent nodal/map checks."""

    destinations: tuple[int, ...]
    region_preserved: bool
    metric_preserved: bool
    capacity_preserved: bool
    operator_checks: tuple[tuple[str, ExactSymmetryDefect], ...]
    field_checks: tuple[tuple[str, ExactSymmetryDefect], ...]


@dataclass(frozen=True)
class ExactMapSymmetry:
    """Complete finite support action and conditional map-symmetry subgroups.

    Operators and fields are supplied independently. The common operator group
    preserves region, metric, capacity and every declared matrix, but need not
    preserve the fields or a full runtime history. Its fixed environmental
    basis is a condition on inputs, not evidence that native inputs satisfy it.
    Index permutations and orbits refer to the retained node order.
    """

    nodes: tuple
    region_indices: tuple[int, ...]
    metric_weights: Vector
    capacity: Vector
    operators: tuple[tuple[str, Matrix], ...]
    fields: tuple[tuple[str, Vector], ...]
    permutations: tuple[ExactPermutationSymmetry, ...]
    admissible_indices: tuple[int, ...]
    operator_group_indices: tuple
    field_group_indices: tuple
    common_group_indices: tuple[int, ...]
    common_operator_group: tuple[tuple[int, ...], ...]
    common_operator_orbits: tuple
    fixed_input_labels: tuple[str, ...]
    fixed_input_basis: Matrix
    support_group_order: int
    cap: int


def observe_exact_map_symmetries(
    snapshot, *, metric_weights, region, operators, fields=None, cap=2000,
):
    """Audit exact declared maps against all weighted-support automorphisms.

    The detached simple undirected graph includes zero-conductance support
    edges. Enumeration delegates to the shared complete-or-refuse symmetry
    owner. Node attributes, region and positive full metric are checked
    separately; exact support symmetry alone is never promoted to dynamical
    symmetry. This observer executes no nodal flow, operator or pressure kernel.
    ``cap`` bounds enumeration, not a physical coefficient or graph-size law.
    """
    import networkx as nx

    if type(cap) is not int or cap <= 0:
        raise ValueError("cap must be a positive nonboolean integer")
    if not isinstance(snapshot, SupportTransportSnapshot):
        raise TypeError("a detached SupportTransportSnapshot is required")
    checked = _from_data(snapshot.nodes, snapshot.conductance,
                         snapshot.support_neighbors, snapshot.epi,
                         snapshot.capacity, snapshot.stored_pressure)
    if checked != snapshot:
        raise ValueError("snapshot derived fields disagree with its data")
    nodes, support, capacity = checked.nodes, checked.support_neighbors, checked.capacity
    size = len(nodes)
    if size < 2 or any(value <= 0 for value in capacity):
        raise ValueError("positive capacity on at least two nodes is required")
    if any(i not in support[j] for i, row in enumerate(support) for j in row):
        raise ValueError("exact weighted symmetry requires undirected support")
    metric = ordered_vector(metric_weights, "metric weights")
    if len(metric) != size or any(value <= 0 for value in metric):
        raise ValueError("a positive full metric matching the node order is required")
    if isinstance(region, (str, bytes, bytearray, Mapping, Set)):
        raise TypeError("region must be an ordered sequence of node labels")
    region_nodes = tuple(region)
    if (not region_nodes or len(region_nodes) >= size
            or len(set(region_nodes)) != len(region_nodes)
            or any(node not in nodes for node in region_nodes)):
        raise ValueError("region must be a nonempty proper set of distinct nodes")
    region_indices = tuple(nodes.index(node) for node in region_nodes)
    region_set = set(region_indices)
    if not isinstance(operators, Mapping) or not operators:
        raise ValueError("at least one named declared operator matrix is required")
    if fields is None:
        fields = {}
    if not isinstance(fields, Mapping):
        raise TypeError("fields must be a mapping of names to vectors")
    if any(not isinstance(name, str) or not name for name in (*operators, *fields)):
        raise ValueError("operator and field names must be nonempty strings")
    matrices = []
    for name, raw in operators.items():
        if isinstance(raw, (str, bytes, bytearray, Mapping, Set)):
            raise TypeError("operator matrices must have ordered rows")
        matrix = tuple(ordered_vector(row, name) for row in raw)
        if len(matrix) != size or any(len(row) != size for row in matrix):
            raise ValueError("operator matrix dimensions must match the node order")
        matrices.append((name, matrix))
    vectors = tuple((name, ordered_vector(value, name)) for name, value in fields.items())
    if any(len(value) != size for _, value in vectors):
        raise ValueError("field dimensions must match the node order")
    conductance = {(i, j): w for i, j, w in checked.conductance}
    graph = nx.Graph()
    graph.add_nodes_from(range(size))
    for i, row in enumerate(support):
        for j in row:
            if i <= j:
                graph.add_edge(i, j, weight=conductance.get((i, j), F(0)))
    mappings = automorphism_permutations(graph, weight="weight", cap=cap)
    permutations = tuple(sorted(tuple(mapping[i] for i in range(size)) for mapping in mappings))

    def defect(entries):
        witness, maximum = None, F(0)
        for indices, value in entries:
            if value and witness is None:
                witness = (*indices, value)
            maximum = max(maximum, abs(value))
        return ExactSymmetryDefect(witness is None, maximum, witness)

    records = []
    for p in permutations:
        matrix_checks = tuple((name, defect(
            (((i, j), matrix[p[i]][p[j]]-matrix[i][j]) for i in range(size) for j in range(size))))
            for name, matrix in matrices)
        field_checks = tuple((name, defect((((i,), value[p[i]]-value[i]) for i in range(size))))
                             for name, value in vectors)
        records.append(ExactPermutationSymmetry(
            p, {p[i] for i in region_indices} == region_set,
            all(metric[p[i]] == metric[i] for i in range(size)),
            all(capacity[p[i]] == capacity[i] for i in range(size)),
            matrix_checks, field_checks,
        ))
    admitted = tuple(i for i, row in enumerate(records)
                     if row.region_preserved and row.metric_preserved and row.capacity_preserved)
    operator_groups = tuple((name, tuple(i for i in admitted if dict(records[i].operator_checks)[name].preserved))
                            for name, _ in matrices)
    field_groups = tuple((name, tuple(i for i in admitted if dict(records[i].field_checks)[name].preserved))
                         for name, _ in vectors)
    common_indices = tuple(i for i in admitted if all(item.preserved for _, item in records[i].operator_checks))
    common = tuple(permutations[i] for i in common_indices)
    if tuple(range(size)) not in common:
        raise RuntimeError("the complete common group lost its identity")
    orbits = tuple(tuple(orbit) for orbit in automorphism_orbits(
        graph, nodes=tuple(range(size)),
        permutations=[dict(enumerate(p)) for p in common],
    ))
    outside_orbits = tuple(orbit for orbit in orbits if not region_set.intersection(orbit))
    fixed_basis = (tuple(F(i in region_set) for i in range(size)),)
    fixed_basis += tuple(tuple(F(i in orbit) for i in range(size)) for orbit in outside_orbits)
    labels = ("region_constant",)+tuple(f"outside_orbit_{k}" for k in range(len(outside_orbits)))
    for value in fixed_basis:
        if any(value[p[i]] != value[i] for p in common for i in range(size)):
            raise RuntimeError("fixed environmental basis is not group invariant")
    return ExactMapSymmetry(
        nodes, region_indices, metric, capacity, tuple(matrices), vectors,
        tuple(records), admitted, operator_groups, field_groups, common_indices,
        common, orbits, labels, fixed_basis, len(permutations), cap,
    )
