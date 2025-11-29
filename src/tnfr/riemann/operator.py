"""Toy TNFR–Riemann operator prototypes.

This module implements a very small experimental operator ``H_TNFR``
acting on prime-labeled graphs.

The goal is *not* to approximate the true Riemann operator, but to
create a concrete playground aligned with the TNFR–Riemann research
notes where we can study how prime-structured Laplacians plus simple
structural potentials behave spectrally.

Status
------
Experimental, non-canonical. Subject to change or removal at any time.

Design
------
We work with simple undirected graphs whose nodes are labelled by
positive integers, typically primes.  Given such a graph ``G`` and a
potential function ``V(n)`` defined on node labels ``n``, we form

    H_TNFR = L + diag(V),

where ``L`` is the combinatorial Laplacian of ``G``.  Eigenvalues of
``H_TNFR`` can then be inspected numerically.

For convenience and reproducibility we provide a minimal constructor
``build_prime_path_graph`` that creates a path graph on the first ``k``
primes, with optional edge weights derived from log-prime distances.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import networkx as nx


def _first_primes(count: int) -> list[int]:
    """Return the first ``count`` prime numbers.

    This is a tiny helper for experimentation and deliberately
    minimalist; it is *not* optimized for large ``count``.
    """

    if count <= 0:
        return []

    primes: list[int] = []
    n = 2
    while len(primes) < count:
        is_prime = True
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(n)
        n += 1
    return primes


def build_prime_path_graph(
    count: int,
    *,
    weight_by_log_gap: bool = True,
) -> nx.Graph:
    """Build a simple path graph whose nodes are the first ``count`` primes.

    Parameters
    ----------
    count:
        Number of prime nodes to include.
    weight_by_log_gap:
        If ``True``, edges are weighted by the absolute difference
        of the logarithms of consecutive primes.  Otherwise all
        edges have weight 1.

    Returns
    -------
    G:
        A NetworkX :class:`~networkx.Graph` with node attribute
        ``"label"`` storing the prime number for each node.
    """

    primes = _first_primes(count)
    G = nx.Graph()

    for idx, p in enumerate(primes):
        G.add_node(idx, label=p)

    for idx in range(len(primes) - 1):
        p1 = primes[idx]
        p2 = primes[idx + 1]
        if weight_by_log_gap:
            w = abs(np.log(p2) - np.log(p1))
        else:
            w = 1.0
        G.add_edge(idx, idx + 1, weight=float(w))

    return G


def default_prime_potential(label: int, sigma: float = 0.5) -> float:
    """Very simple structural potential on a prime label.

    The form is inspired by the discussion in the TNFR–Riemann notes
    where ``(sigma - 1/2) * log p`` acts as an effective energy term.

    When ``sigma = 0.5`` this potential vanishes, so the operator
    reduces to a pure Laplacian.
    """

    return float((sigma - 0.5) * np.log(float(label)))


def build_h_tnfr(
    G: nx.Graph,
    *,
    sigma: float = 0.5,
    potential_fn: Callable[[int, float], float] = default_prime_potential,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct a matrix representation of the toy ``H_TNFR`` operator.

    Parameters
    ----------
    G:
        Undirected graph whose nodes carry an integer ``"label"``
        attribute, typically a prime.
    sigma:
        Real parameter analogous to Re(s) in zeta.  When ``sigma`` is
        exactly 0.5 the potential term is identically zero and the
        operator is the combinatorial Laplacian.
    potential_fn:
        Function mapping ``(label, sigma)`` to a real potential value.

    Returns
    -------
    H, diag_V:
        A tuple where ``H`` is the dense matrix representation of
        ``H_TNFR`` and ``diag_V`` is the diagonal potential matrix for
        inspection.
    """

    if not isinstance(G, nx.Graph):
        raise TypeError("G must be an undirected networkx.Graph")

    # Ensure deterministic node ordering
    nodes = sorted(G.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    L: np.ndarray = np.zeros((n, n), dtype=float)

    # Build weighted Laplacian L = D - A
    for u, v, data in G.edges(data=True):
        i = index[u]
        j = index[v]
        w = float(data.get("weight", 1.0))
        L[i, j] -= w
        L[j, i] -= w
        L[i, i] += w
        L[j, j] += w

    diag_V: np.ndarray = np.zeros((n, n), dtype=float)
    for node in nodes:
        i = index[node]
        label = G.nodes[node].get("label")
        if label is None:
            raise ValueError("All nodes must have an integer 'label' attribute")
        v_val = float(potential_fn(int(label), sigma))
        diag_V[i, i] = v_val

    H = L + diag_V
    return H, diag_V
