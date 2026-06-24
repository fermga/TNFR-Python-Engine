"""Regression tests for canonical-field cache invalidation.

These tests pin a real correctness bug: ``@cache_tnfr_computation`` builds its
cache key from a dependency hash (:func:`tnfr.utils.cache._compute_dependency_hash`).
That hash used hardcoded English attribute names (``'delta_nfr'``, ``'vf'``,
``'epi'``) to read node values, but the canonical writer
(:func:`tnfr.alias.set_attr`) stores them under the FIRST alias of each field —
the Greek/canonical key (``'ΔNFR'``, ``'νf'``, ``'EPI'``).  The mismatch made the
dependency hash read ``None`` for every node, so the cache key was BLIND to the
field: changing ΔNFR on a fixed topology returned stale Φ_s / ξ_C / J_ΔNFR, and
two distinct graphs with identical topology but different fields collided.

The fix resolves dependencies through the canonical alias tuples
(:func:`tnfr.utils.cache._dependency_alias_keys`).  These tests fail on the old
code and pass on the fixed code.
"""

from __future__ import annotations

import math
import random

import networkx as nx
import numpy as np

from tnfr.alias import get_attr, set_attr
from tnfr.constants.aliases import ALIAS_DNFR, ALIAS_EPI, ALIAS_VF
from tnfr.dynamics import default_compute_delta_nfr
from tnfr.physics.canonical import (
    compute_structural_potential,
    estimate_coherence_length,
)
from tnfr.utils.cache import _compute_dependency_hash


def _build(n: int = 80, seed: int = 7) -> nx.Graph:
    rng = random.Random(seed)
    G = nx.watts_strogatz_graph(n, 6, 0.2, seed=seed)
    for nd in G.nodes():
        G.nodes[nd]["theta"] = rng.uniform(0.0, 2.0 * math.pi)
        set_attr(G.nodes[nd], ALIAS_EPI, rng.uniform(-0.35, 0.35))
        set_attr(G.nodes[nd], ALIAS_VF, 1.0)
    default_compute_delta_nfr(G)
    return G


def test_dependency_hash_reflects_dnfr_change():
    """The dependency hash must change when ΔNFR changes (node_dnfr)."""
    G = _build()
    deps = {"graph_topology", "node_dnfr"}
    h1 = _compute_dependency_hash(G, deps)
    for nd in G.nodes():
        set_attr(
            G.nodes[nd],
            ALIAS_DNFR,
            float(get_attr(G.nodes[nd], ALIAS_DNFR, 0.0)) * 5.0 + 1.0,
        )
    h2 = _compute_dependency_hash(G, deps)
    assert h1 != "", "dependency hash should be non-empty when fields present"
    assert h1 != h2, "node_dnfr dependency hash must change when ΔNFR changes"


def test_dependency_hash_reflects_vf_change():
    """The dependency hash must change when νf changes (node_vf)."""
    G = _build()
    deps = {"graph_topology", "node_vf"}
    h1 = _compute_dependency_hash(G, deps)
    for i, nd in enumerate(G.nodes()):
        set_attr(G.nodes[nd], ALIAS_VF, 1.0 + 0.5 * (i % 3))
    h2 = _compute_dependency_hash(G, deps)
    assert h1 != h2, "node_vf dependency hash must change when νf changes"


def test_dependency_hash_reflects_epi_change():
    """The dependency hash must change when EPI changes (node_epi)."""
    G = _build()
    deps = {"graph_topology", "node_epi"}
    h1 = _compute_dependency_hash(G, deps)
    for nd in G.nodes():
        set_attr(
            G.nodes[nd], ALIAS_EPI, float(get_attr(G.nodes[nd], ALIAS_EPI, 0.0)) + 2.0
        )
    h2 = _compute_dependency_hash(G, deps)
    assert h1 != h2, "node_epi dependency hash must change when EPI changes"


def test_structural_potential_responds_to_dnfr_change():
    """Φ_s must recompute (not return stale cache) after an in-place ΔNFR change."""
    G = _build()
    v1 = np.array([compute_structural_potential(G)[k] for k in sorted(G.nodes())])
    for nd in G.nodes():
        set_attr(
            G.nodes[nd],
            ALIAS_DNFR,
            float(get_attr(G.nodes[nd], ALIAS_DNFR, 0.0)) * 5.0 + 1.0,
        )
    v2 = np.array([compute_structural_potential(G)[k] for k in sorted(G.nodes())])
    assert (
        float(np.max(np.abs(v2 - v1))) > 1e-6
    ), "Φ_s returned a stale cached value after ΔNFR changed"


def test_structural_potential_no_collision_same_topology():
    """Two graphs with identical topology but different ΔNFR must not collide."""

    def mk(scale: float) -> nx.Graph:
        rng = random.Random(99)
        H = nx.watts_strogatz_graph(60, 6, 0.2, seed=99)
        for nd in H.nodes():
            H.nodes[nd]["theta"] = rng.uniform(0.0, 2.0 * math.pi)
            set_attr(H.nodes[nd], ALIAS_EPI, rng.uniform(-0.35, 0.35))
            set_attr(H.nodes[nd], ALIAS_VF, 1.0)
        default_compute_delta_nfr(H)
        for nd in H.nodes():
            set_attr(
                H.nodes[nd],
                ALIAS_DNFR,
                float(get_attr(H.nodes[nd], ALIAS_DNFR, 0.0)) * scale,
            )
        return H

    a = mk(1.0)
    b = mk(3.0)
    pa = np.array([compute_structural_potential(a)[k] for k in sorted(a.nodes())])
    pb = np.array([compute_structural_potential(b)[k] for k in sorted(b.nodes())])
    assert (
        float(np.max(np.abs(pa - pb))) > 1e-6
    ), "Φ_s collided across two same-topology graphs with different ΔNFR"


def test_coherence_length_responds_to_dnfr_change():
    """ξ_C (also node_dnfr-dependent) must not return a stale cached value."""
    G = _build()
    xi1 = estimate_coherence_length(G)
    for nd in G.nodes():
        set_attr(
            G.nodes[nd],
            ALIAS_DNFR,
            float(get_attr(G.nodes[nd], ALIAS_DNFR, 0.0)) * 7.0 - 2.0,
        )
    xi2 = estimate_coherence_length(G)
    # Either the value changes, or both are non-finite/degenerate; the bug
    # produced bit-identical finite values regardless of ΔNFR.
    if math.isfinite(xi1) and math.isfinite(xi2):
        assert (
            abs(xi2 - xi1) > 1e-9
        ), "ξ_C returned a stale cached value after ΔNFR changed"
