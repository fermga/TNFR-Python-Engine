r"""Symmetry-sector decomposition for the canonical emergent operator (R1).

For any graph ``G`` with automorphism group ``Γ = Aut(G, W)``, the canonical
structural-diffusion operator ``L_rw = I − D⁻¹W`` is **equivariant**: it commutes
with the permutation representation of every automorphism, ``P_σ L_rw =
L_rw P_σ``.  By Schur's lemma an equivariant operator block-diagonalises by the
isotypic components of ``Γ``; the coarsest split is

    ℝᴺ = Fix(Γ) ⊕ Fix(Γ)^⊥,

where ``Fix(Γ)`` (functions constant on the orbits of ``Γ``) is the image of the
**Reynolds projector** ``Q_Γ = (1/|Γ|) Σ_σ P_σ`` and ``dim Fix(Γ)`` equals the
number of vertex orbits.  Any per-node observable that is itself Γ-invariant
lands in ``Fix(Γ)`` — it can tell orbit from orbit, never node from node within
an orbit; all node-separating information lives in ``Fix(Γ)^⊥`` (the spectrum).

This module supplies the orbit partition, the Reynolds projector and the sector
decomposition; :mod:`tnfr.physics.equivariance` certifies the equivariance of
``L_rw`` and its preservation of the sectors.  It formalises the structure
measured in example 123.  A localized (single-node) emission is *not*
equivariant: choosing an origin ``o`` turns ``G`` into a pointed graph ``(G, o)``
and reduces the relevant group to the stabilizer of ``o`` — the symmetry break is
then declared, not spontaneous.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "automorphism_permutations",
    "permutation_matrix",
    "automorphism_orbits",
    "orbit_count",
    "reynolds_projector",
    "decompose_state",
    "is_orbit_constant",
]


def _node_list(G, nodes) -> list:
    return list(G.nodes()) if nodes is None else list(nodes)


def automorphism_permutations(G, *, weight: str | None = None, cap: int = 2000):
    r"""Automorphisms of ``G`` as node→node maps (weight/direction aware).

    Uses the VF2 (di)graph matcher.  When ``weight`` is given, only automorphisms
    that preserve that edge attribute are returned; for a directed ``G`` the
    directed matcher preserves edge orientation.  At most ``cap`` maps.
    """
    from networkx.algorithms import isomorphism as iso

    matcher_cls = iso.DiGraphMatcher if G.is_directed() else iso.GraphMatcher
    kwargs = {}
    if weight is not None:
        kwargs["edge_match"] = iso.numerical_edge_match(weight, 1.0)
    matcher = matcher_cls(G, G, **kwargs)
    out: list[dict] = []
    for mapping in matcher.isomorphisms_iter():
        out.append(dict(mapping))
        if len(out) >= cap:
            break
    return out


def permutation_matrix(mapping: dict, nodes) -> np.ndarray:
    r"""Permutation matrix ``P`` with ``P[dst, src] = 1`` for ``src → dst``."""
    idx = {nd: i for i, nd in enumerate(nodes)}
    n = len(idx)
    P = np.zeros((n, n))
    for src, dst in mapping.items():
        P[idx[dst], idx[src]] = 1.0
    return P


def automorphism_orbits(
    G,
    *,
    nodes=None,
    permutations=None,
    weight: str | None = None,
    cap: int = 2000,
) -> list[tuple]:
    r"""Vertex orbits of ``Aut(G)`` (union-find over the automorphisms)."""
    nodes = _node_list(G, nodes)
    perms = (
        permutations
        if permutations is not None
        else automorphism_permutations(G, weight=weight, cap=cap)
    )
    idx = {nd: i for i, nd in enumerate(nodes)}
    parent = list(range(len(nodes)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for mapping in perms:
        for src, dst in mapping.items():
            ra, rb = find(idx[src]), find(idx[dst])
            if ra != rb:
                parent[ra] = rb
    groups: dict[int, list] = {}
    for nd in nodes:
        groups.setdefault(find(idx[nd]), []).append(nd)
    return [tuple(members) for members in groups.values()]


def orbit_count(G, **kwargs) -> int:
    r"""Number of vertex orbits of ``Aut(G)`` (``= dim Fix(Γ)``)."""
    return len(automorphism_orbits(G, **kwargs))


def reynolds_projector(
    G,
    *,
    nodes=None,
    permutations=None,
    weight: str | None = None,
    cap: int = 2000,
) -> np.ndarray:
    r"""Reynolds projector ``Q_Γ = (1/|Γ|) Σ_σ P_σ`` onto ``Fix(Γ)``."""
    nodes = _node_list(G, nodes)
    perms = (
        permutations
        if permutations is not None
        else automorphism_permutations(G, weight=weight, cap=cap)
    )
    n = len(nodes)
    P = np.zeros((n, n))
    for mapping in perms:
        P += permutation_matrix(mapping, nodes)
    return P / len(perms)


def _as_vector(field, nodes) -> np.ndarray:
    if isinstance(field, dict):
        return np.array([float(field[nd]) for nd in nodes], dtype=float)
    return np.asarray(field, dtype=float)


def decompose_state(
    G,
    field,
    *,
    nodes=None,
    permutations=None,
    weight: str | None = None,
    cap: int = 2000,
):
    r"""Split a per-node field into ``(Fix(Γ), Fix(Γ)^⊥)`` components."""
    nodes = _node_list(G, nodes)
    Q = reynolds_projector(
        G, nodes=nodes, permutations=permutations, weight=weight, cap=cap
    )
    v = _as_vector(field, nodes)
    fixed = Q @ v
    return fixed, v - fixed


def is_orbit_constant(
    G,
    values,
    *,
    tol: float = 1e-9,
    nodes=None,
    permutations=None,
    weight: str | None = None,
    cap: int = 2000,
) -> bool:
    r"""Whether a per-node field is constant on orbits (lands in ``Fix(Γ)``).

    An orbit-constant ``νf`` keeps the nodal-equation flow inside the Reynolds
    sectors; a non-orbit-constant per-node lever breaks the symmetry externally.
    """
    nodes = _node_list(G, nodes)
    v = _as_vector(values, nodes)
    Q = reynolds_projector(
        G, nodes=nodes, permutations=permutations, weight=weight, cap=cap
    )
    return bool(np.linalg.norm(v - Q @ v) <= tol)
