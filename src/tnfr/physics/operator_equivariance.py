r"""Operator-by-operator equivariance audit (R1, non-linear stage).

The diffusion base case (:mod:`tnfr.physics.equivariance`) proves ``L_rw`` is
equivariant.  The symmetry-sector theorem for full TNFR *words* additionally
needs each operator ``O`` to be equivariant: ``O(P_σ x) = P_σ O(x)`` for every
automorphism ``σ``.  The testable consequence is that an equivariant operator
maps a symmetric (``Fix(Γ)``) state to a symmetric state — applying it at a node
``v`` and at ``σ(v)`` on a σ-invariant seed yields σ-related results.

This module measures that residual per operator.  A non-zero residual does not
break the framework: it *localizes* where symmetry breaking enters (a label- or
selection-dependent action), which is the falsification clause of the programme
(report §8.1.7).  A localized single-node **emission** is pointed by choice of
origin; here we audit the operator's per-node *action*, not the choice of origin.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from ..alias import get_attr, set_attr
from ..constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_THETA, ALIAS_VF

__all__ = [
    "OperatorEquivarianceResult",
    "operator_equivariance_residual",
    "audit_operator_equivariance",
]

_CHANNELS = (ALIAS_EPI, ALIAS_THETA, ALIAS_VF, ALIAS_DNFR)

# TNFR content-keyed caches live in ``G.graph`` and survive ``networkx``'s
# ``G.copy()`` by shared reference. They are keyed on a **label-independent**
# node-set checksum, so on a symmetric graph two isomorphic copies (``B = σ(A)``)
# collide and ``B`` reads ``A``'s cached ΔNFR prep — the cross-operator leak
# (report R1-T01). Dropping these containers per copy makes each measurement an
# independent experiment with a clean, private cache; config keys
# (``_dnfr_weights``, ``_DNFR_META``, ``_dnfr_hook_name``) are preserved.
_GRAPH_CACHE_HINTS = (
    "cache", "checksum", "dirty", "_node_list", "_node_set",
)


def _isolate_graph_caches(H) -> None:
    """Drop the TNFR content-keyed caches ``H`` inherited via ``G.copy()``.

    After this ``H`` recomputes ΔNFR from a clean, private cache, so a warm
    (cross-experiment) engine cache cannot leak an isomorphic sibling's value.
    """
    stale = [
        key for key in list(H.graph)
        if any(hint in key for hint in _GRAPH_CACHE_HINTS)
    ]
    for key in stale:
        H.graph.pop(key, None)


def operator_equivariance_residual(op, G, sigma: dict, node, *, channels=None):
    r"""Residual ``max_u,ch |A[u] − B[σ(u)]|`` for ``O@node`` vs ``O@σ(node)``.

    ``A`` applies ``op`` at ``node``; ``B`` applies it at ``σ(node)``.  If ``op``
    is equivariant and the seed is σ-invariant, ``O@σ(node) = σ(O@node)`` so
    ``B[σ(u)] = A[u]`` for every node ``u`` and channel.  ΔNFR is recomputed on
    both (it is an emergent network field).  Each copy's inherited content-keyed
    graph caches are isolated (:func:`_isolate_graph_caches`), so the result is
    independent of any warm engine cache — an isomorphic sibling cannot leak its
    ΔNFR prep across the isomorphism.
    """
    from ..dynamics import default_compute_delta_nfr

    channels = _CHANNELS if channels is None else channels
    A = G.copy()
    B = G.copy()
    _isolate_graph_caches(A)  # private, clean cache per copy (no sibling leak)
    _isolate_graph_caches(B)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        op(A, node)
        op(B, sigma[node])
    default_compute_delta_nfr(A)
    default_compute_delta_nfr(B)
    resid = 0.0
    for u in G.nodes():
        su = sigma[u]
        for ch in channels:
            a = float(get_attr(A.nodes[u], ch, 0.0))
            b = float(get_attr(B.nodes[su], ch, 0.0))
            resid = max(resid, abs(a - b))
    return resid


@dataclass(frozen=True)
class OperatorEquivarianceResult:
    r"""Per-operator equivariance verdict on the audit test cases."""

    name: str
    glyph: str
    residual: float
    is_equivariant: bool


def _seed(G, theta_of, epi_of, vf_of):
    from ..dynamics import default_compute_delta_nfr

    for nd in G.nodes():
        epi = float(epi_of(nd))
        set_attr(G.nodes[nd], ALIAS_THETA, float(theta_of(nd)))
        set_attr(G.nodes[nd], ALIAS_EPI, epi)
        set_attr(G.nodes[nd], ALIAS_VF, float(vf_of(nd)))
        # The common fixture executes every canonical operator. ZHIR's
        # non-disableable trigger therefore needs a replayable, signed-growth
        # witness anchored to the live EPI value.
        G.nodes[nd]["epi_history"] = [epi - 1.0, epi]
    default_compute_delta_nfr(G)
    return G


def _test_cases():
    r"""Two σ-invariant seeds: a vertex-transitive cycle (rotation) and a star
    with two orbits (leaf swap), the latter with an orbit-constant phase so that
    Coupling/Resonance are exercised non-trivially."""
    import networkx as nx

    # Cycle C6, rotation i -> i+1 (uniform seed lies in Fix(Gamma)).
    cyc = nx.cycle_graph(6)
    _seed(cyc, lambda n: 0.2, lambda n: 0.4, lambda n: 1.0)
    cyc_sigma = {i: (i + 1) % 6 for i in range(6)}

    # Star K1,4 (0=center, 1..4 leaves), swap leaves 1<->2; orbit-constant seed
    # (center phase != leaf phase, within the U3 gate |Δφ| <= pi/2).
    star = nx.star_graph(4)
    _seed(
        star,
        lambda n: 0.1 if n == 0 else 0.3,
        lambda n: 0.5 if n == 0 else 0.3,
        lambda n: 1.0,
    )
    star_sigma = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4}

    return [(cyc, cyc_sigma, 0), (star, star_sigma, 1)]


def audit_operator_equivariance(*, tol: float = 1e-6):
    r"""Audit all 13 canonical operators for Fix(Γ)-preserving equivariance.

    Returns one :class:`OperatorEquivarianceResult` per operator, reporting the
    worst residual over the test cases and whether it is within ``tol``.
    """
    from ..operators.definitions import (
        Coherence,
        Contraction,
        Coupling,
        Dissonance,
        Emission,
        Expansion,
        Mutation,
        Reception,
        Recursivity,
        Resonance,
        SelfOrganization,
        Silence,
        Transition,
    )

    catalog = [
        ("emission", "AL", Emission),
        ("reception", "EN", Reception),
        ("coherence", "IL", Coherence),
        ("dissonance", "OZ", Dissonance),
        ("coupling", "UM", Coupling),
        ("resonance", "RA", Resonance),
        ("silence", "SHA", Silence),
        ("expansion", "VAL", Expansion),
        ("contraction", "NUL", Contraction),
        ("self_organization", "THOL", SelfOrganization),
        ("mutation", "ZHIR", Mutation),
        ("transition", "NAV", Transition),
        ("recursivity", "REMESH", Recursivity),
    ]
    cases = _test_cases()
    results: list[OperatorEquivarianceResult] = []
    for name, glyph, cls in catalog:
        worst = 0.0
        for G, sigma, node in cases:
            op = cls()
            try:
                r = operator_equivariance_residual(op, G, sigma, node)
            except Exception:
                r = float("inf")
            worst = max(worst, r)
        results.append(
            OperatorEquivarianceResult(name, glyph, worst, worst < tol)
        )
    return results
