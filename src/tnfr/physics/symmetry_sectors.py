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
decomposition. Automatic enumeration returns the complete automorphism group
of a simple graph, or raises when its cap is exceeded. Explicitly supplied
complete bijections instead define the group they generate; they need not list
every group element and are not authenticated against graph attributes.
The projector averages within generated orbits, rather than averaging an
incomplete list of permutations. Its entries are represented in binary64.

:mod:`tnfr.physics.equivariance` measures diffusion commutation separately.
These finite graph/vertex-action diagnostics do not authenticate the full nodal
state, histories, runtime operators or birth support. A declared origin ``o``
turns ``G`` into a pointed graph ``(G, o)`` and reduces the graph symmetry group
to its stabilizer; a particular operator need not break every other symmetry.
"""

from __future__ import annotations

from collections.abc import Mapping, Set

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


def _validated_node_order(nodes, *, expected=None) -> tuple[list, dict]:
    """Materialize one complete, duplicate-free ordered vertex domain."""
    if isinstance(nodes, (str, bytes)) or (
        isinstance(nodes, Set) and not isinstance(nodes, Mapping)
    ):
        raise ValueError("nodes must be an ordered sequence of distinct vertices")
    try:
        ordered = list(nodes)
        index = {node: i for i, node in enumerate(ordered)}
    except TypeError as exc:
        raise ValueError("nodes must contain hashable vertices") from exc
    if len(index) != len(ordered):
        raise ValueError("nodes must contain distinct vertices")
    if expected is not None and set(index) != set(expected):
        raise ValueError("node order must contain every graph vertex exactly once")
    return ordered, index


def _node_list(G, nodes) -> list:
    ordered = list(G.nodes()) if nodes is None else nodes
    return _validated_node_order(ordered, expected=G.nodes())[0]


def _validated_bijection(mapping, index) -> tuple[int, ...]:
    """Return destination indices only after validating the complete map."""
    if not isinstance(mapping, Mapping):
        raise ValueError("each permutation must be a complete vertex mapping")
    try:
        if set(mapping) != set(index):
            raise ValueError("permutation domain must equal the complete node order")
        destinations = tuple(mapping[node] for node in index)
        if set(destinations) != set(index):
            raise ValueError("permutation must be a bijection of the complete node order")
        return tuple(index[node] for node in destinations)
    except TypeError as exc:
        raise ValueError("permutation destinations must be hashable vertices") from exc


def automorphism_permutations(G, *, weight: str | None = None, cap: int = 2000):
    r"""Enumerate all simple-graph automorphisms, or reject cap exhaustion.

    The VF2 (di)graph matcher preserves direction and, when requested, exact
    edge-attribute equality (missing values mean 1.0). No numeric tolerance is
    used. Nonreflexive or nonscalar equality is rejected. Parallel-edge graphs
    are unsupported: this function declares no multiedge matching convention.
    ``cap`` is a positive nonboolean integer; observing map ``cap+1`` raises
    without returning a truncated group. Node attributes are not compared.
    """
    from networkx.algorithms import isomorphism as iso

    if type(cap) is not int or cap <= 0:
        raise ValueError("cap must be a positive nonboolean integer")
    if G.is_multigraph():
        raise ValueError("automatic automorphisms support simple graphs, not multigraphs")
    if weight is not None and not isinstance(weight, str):
        raise ValueError("weight must be an edge-attribute name or None")
    matcher_cls = iso.DiGraphMatcher if G.is_directed() else iso.GraphMatcher
    kwargs = {}
    if weight is not None:
        for _, _, data in G.edges(data=True):
            value = data.get(weight, 1.0)
            equal = value == value
            if not isinstance(equal, (bool, np.bool_)) or not equal:
                raise ValueError("edge attributes require reflexive scalar equality")

        def exact_edge_match(left, right):
            equal = left.get(weight, 1.0) == right.get(weight, 1.0)
            if not isinstance(equal, (bool, np.bool_)):
                raise ValueError("edge attributes require scalar equality")
            return bool(equal)

        kwargs["edge_match"] = exact_edge_match
    matcher = matcher_cls(G, G, **kwargs)
    out: list[dict] = []
    for mapping in matcher.isomorphisms_iter():
        if len(out) == cap:
            raise ValueError("automorphism group exceeds cap; no partial group is returned")
        out.append(dict(mapping))
    return out


def permutation_matrix(mapping: dict, nodes) -> np.ndarray:
    r"""Matrix ``P[dst, src] = 1`` for a validated complete vertex bijection.

    This authenticates a permutation of the supplied ordered domain, not a
    graph automorphism or a symmetry of a complete runtime state.
    """
    nodes, idx = _validated_node_order(nodes)
    destinations = _validated_bijection(mapping, idx)
    n = len(nodes)
    P = np.zeros((n, n))
    for src, dst in enumerate(destinations):
        P[dst, src] = 1.0
    return P


def automorphism_orbits(
    G,
    *,
    nodes=None,
    permutations=None,
    weight: str | None = None,
    cap: int = 2000,
) -> list[tuple]:
    r"""Orbits of complete ``Aut(G)`` or the supplied generated vertex group.

    Explicit ``permutations`` must be complete bijections on all graph nodes.
    Their generated group is used without enumerating its closure; duplicates
    are harmless and an empty list generates the identity group. Supplied
    maps need not preserve graph support, weights or nodal state. ``nodes``
    may reorder, but cannot omit or duplicate, graph vertices. ``weight`` and
    ``cap`` configure automatic enumeration only.
    """
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
        for src, dst in enumerate(_validated_bijection(mapping, idx)):
            ra, rb = find(src), find(dst)
            if ra != rb:
                parent[ra] = rb
    groups: dict[int, list] = {}
    for nd in nodes:
        groups.setdefault(find(idx[nd]), []).append(nd)
    return [tuple(members) for members in groups.values()]


def orbit_count(G, **kwargs) -> int:
    r"""Number of orbits of the selected action (``= dim Fix(Γ)``)."""
    return len(automorphism_orbits(G, **kwargs))


def reynolds_projector(
    G,
    *,
    nodes=None,
    permutations=None,
    weight: str | None = None,
    cap: int = 2000,
) -> np.ndarray:
    r"""Orthogonal projector onto fields constant on the selected group orbits.

    Each orbit block is filled with ``1/len(orbit)``. This equals the Reynolds
    average over the complete generated group, even when supplied permutations
    are only generators, repeated or empty. No closure enumeration is needed.
    Explicit maps authenticate only a vertex action; automatic enumeration
    authenticates the selected simple-graph support/edge-attribute symmetry.
    """
    nodes = _node_list(G, nodes)
    orbits = automorphism_orbits(
        G, nodes=nodes, permutations=permutations, weight=weight, cap=cap,
    )
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    P = np.zeros((n, n))
    for orbit in orbits:
        indices = [idx[node] for node in orbit]
        P[np.ix_(indices, indices)] = 1.0 / len(orbit)
    return P


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

    For the isolated fixed EPI channel and a group preserving its weighted
    support, an orbit-constant ``νf`` preserves the generator's symmetry.
    This does not authenticate other pressure channels or supplied maps as
    graph symmetries; a non-orbit-constant capacity can break that symmetry.
    """
    nodes = _node_list(G, nodes)
    v = _as_vector(values, nodes)
    Q = reynolds_projector(
        G, nodes=nodes, permutations=permutations, weight=weight, cap=cap
    )
    return bool(np.linalg.norm(v - Q @ v) <= tol)
