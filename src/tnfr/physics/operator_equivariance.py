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


def _clear_module_caches() -> None:
    """Best-effort reset of content-keyed module caches between operators.

    The engine memoizes per-node work under content digests; on a symmetric
    (content-identical) graph that state can leak between successive operator
    audits and masquerade as non-equivariance.  A single operator measured from
    clean state (e.g. one pytest case under the autouse reset fixture) is exact;
    :func:`audit_operator_equivariance` calls this between operators so the
    standalone batch is best-effort isolated.
    """
    try:
        from ..utils.cache import clear_node_repr_cache, reset_global_cache

        clear_node_repr_cache()
        reset_global_cache()
    except Exception:
        pass


def operator_equivariance_residual(op, G, sigma: dict, node, *, channels=None):
    r"""Residual ``max_u,ch |A[u] − B[σ(u)]|`` for ``O@node`` vs ``O@σ(node)``.

    ``A`` applies ``op`` at ``node``; ``B`` applies it at ``σ(node)``.  If ``op``
    is equivariant and the seed is σ-invariant, ``O@σ(node) = σ(O@node)`` so
    ``B[σ(u)] = A[u]`` for every node ``u`` and channel.  ΔNFR is recomputed on
    both (it is an emergent network field).  Measure one operator from clean
    module state (the engine's content caches leak across operators on a
    symmetric graph).
    """
    from ..dynamics import default_compute_delta_nfr

    channels = _CHANNELS if channels is None else channels
    A = G.copy()
    B = G.copy()
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
        set_attr(G.nodes[nd], ALIAS_THETA, float(theta_of(nd)))
        set_attr(G.nodes[nd], ALIAS_EPI, float(epi_of(nd)))
        set_attr(G.nodes[nd], ALIAS_VF, float(vf_of(nd)))
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
        _clear_module_caches()  # best-effort isolation from the prior operator
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
