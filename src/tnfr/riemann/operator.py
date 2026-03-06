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

from typing import Callable

from ..errors import TNFRValueError
from ..mathematics.unified_numerical import np
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
) -> tuple[np.ndarray, np.ndarray]:
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
            raise TNFRValueError(
                "All nodes must have an integer 'label' attribute",
                context={"node": node},
                suggestion="Assign integer labels to all nodes."
            )
        v_val = float(potential_fn(int(label), sigma))
        diag_V[i, i] = v_val

    H = L + diag_V
    return H, diag_V

def build_prime_cycle_graph(
    count: int,
    *,
    weight_by_log_gap: bool = True,
) -> nx.Graph:
    """Build a cycle graph on the first ``count`` primes.

    Like :func:`build_prime_path_graph` but with an extra edge
    connecting the last prime back to the first (periodic boundary).

    Parameters
    ----------
    count:
        Number of prime nodes (>= 3 for a meaningful cycle).
    weight_by_log_gap:
        If True, edge weights are |log p_{j+1} - log p_j|.
    """
    G = build_prime_path_graph(count, weight_by_log_gap=weight_by_log_gap)
    if count >= 3:
        primes = _first_primes(count)
        if weight_by_log_gap:
            w = abs(np.log(float(primes[-1])) - np.log(float(primes[0])))
        else:
            w = 1.0
        G.add_edge(0, count - 1, weight=float(w))
    return G

def build_prime_star_graph(
    count: int,
    *,
    weight_by_log_gap: bool = True,
) -> nx.Graph:
    """Build a star graph on the first ``count`` primes.

    Node 0 (p=2) is the hub connected to all other nodes.
    Edge weight (0, i) = |log p_i - log 2|.

    Parameters
    ----------
    count:
        Number of prime nodes (>= 2).
    weight_by_log_gap:
        If True, edge weights are |log p_i - log p_0|.
    """
    primes = _first_primes(count)
    G = nx.Graph()
    for idx, p in enumerate(primes):
        G.add_node(idx, label=p)
    for idx in range(1, len(primes)):
        if weight_by_log_gap:
            w = abs(np.log(float(primes[idx])) - np.log(float(primes[0])))
        else:
            w = 1.0
        G.add_edge(0, idx, weight=float(w))
    return G

def build_prime_complete_graph(
    count: int,
    *,
    weight_by_log_gap: bool = True,
) -> nx.Graph:
    r"""Build a complete graph :math:`K_k` on the first ``count`` primes.

    Every pair of primes is connected.  Edge weight (i, j) =
    |log p_i - log p_j|.

    Parameters
    ----------
    count:
        Number of prime nodes.
    weight_by_log_gap:
        If True, edge weights are |log p_i - log p_j|.
    """
    primes = _first_primes(count)
    G = nx.Graph()
    for idx, p in enumerate(primes):
        G.add_node(idx, label=p)
    for i in range(len(primes)):
        for j in range(i + 1, len(primes)):
            if weight_by_log_gap:
                w = abs(np.log(float(primes[j])) - np.log(float(primes[i])))
            else:
                w = 1.0
            G.add_edge(i, j, weight=float(w))
    return G

def build_prime_tree_graph(
    count: int,
    *,
    weight_by_log_gap: bool = True,
) -> nx.Graph:
    """Build a balanced binary tree on the first ``count`` primes.

    Primes are assigned in breadth-first order: p_1 is root, p_2 and
    p_3 are its children, p_4..p_7 at the next level, etc.  Edge
    weight parent-child = |log p_parent - log p_child|.

    Parameters
    ----------
    count:
        Number of prime nodes.
    weight_by_log_gap:
        If True, edge weights are |log p_parent - log p_child|.
    """
    primes = _first_primes(count)
    G = nx.Graph()
    for idx, p in enumerate(primes):
        G.add_node(idx, label=p)
    # BFS-order binary tree: children of node i are 2i+1 and 2i+2
    for i in range(len(primes)):
        left = 2 * i + 1
        right = 2 * i + 2
        for child in (left, right):
            if child < len(primes):
                if weight_by_log_gap:
                    w = abs(np.log(float(primes[child]))
                            - np.log(float(primes[i])))
                else:
                    w = 1.0
                G.add_edge(i, child, weight=float(w))
    return G

def build_prime_random_graph(
    count: int,
    *,
    edge_prob: float = 0.3,
    seed: int = 42,
    weight_by_log_gap: bool = True,
) -> nx.Graph:
    r"""Build an Erdos-Renyi random graph on the first ``count`` primes.

    Each edge (i, j) is included with probability ``edge_prob``.
    If the result is disconnected, edges are added to the largest
    component boundary until the graph is connected.

    Parameters
    ----------
    count:
        Number of prime nodes.
    edge_prob:
        Probability of each edge (0 < p <= 1).
    seed:
        Random seed for reproducibility (Invariant #6).
    weight_by_log_gap:
        If True, edge weights are |log p_i - log p_j|.
    """
    primes = _first_primes(count)
    rng = np.random.RandomState(seed)

    G = nx.Graph()
    for idx, p in enumerate(primes):
        G.add_node(idx, label=p)

    # Add edges with probability edge_prob
    for i in range(len(primes)):
        for j in range(i + 1, len(primes)):
            if rng.random() < edge_prob:
                if weight_by_log_gap:
                    w = abs(np.log(float(primes[j]))
                            - np.log(float(primes[i])))
                else:
                    w = 1.0
                G.add_edge(i, j, weight=float(w))

    # Ensure connectivity: add minimum edges to connect components
    if count >= 2 and not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for ci in range(1, len(components)):
            # Connect component ci to component 0
            u = min(components[0])
            v = min(components[ci])
            if weight_by_log_gap:
                w = abs(np.log(float(primes[v]))
                        - np.log(float(primes[u])))
            else:
                w = 1.0
            G.add_edge(u, v, weight=float(w))
            components[0] = components[0] | components[ci]

    return G

def build_tridiagonal_h_tnfr(
    count: int,
    sigma: float = 0.5,
    *,
    weight_by_log_gap: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Build tridiagonal representation of H_TNFR for a prime path graph.

    Exploits the path graph structure: both L_k (Laplacian of a path)
    and V_sigma (diagonal) are tridiagonal, so H = L + V is tridiagonal.
    This avoids O(k^2) dense matrix construction and enables O(k^2)
    eigenvalue computation via ``scipy.linalg.eigh_tridiagonal``.

    The operator is:

        H^{(k)}(\sigma) = L_k + (\sigma - 1/2) \, \mathrm{diag}(\log p_i)

    Parameters
    ----------
    count : int
        Number of prime nodes (k).
    sigma : float
        Structural parameter. At sigma = 0.5, V = 0 and H = L.
    weight_by_log_gap : bool
        If True, edge weights are |log(p_{i+1}) - log(p_i)|.

    Returns
    -------
    (d, e, log_primes)
        d : main diagonal of H, shape (count,).
        e : sub-diagonal of H, shape (count - 1,).
        log_primes : log(p_i) vector, shape (count,).
    """
    primes = _first_primes(count)
    log_p = np.array([np.log(float(p)) for p in primes])
    k = len(primes)

    if k < 2:
        d = np.zeros(k)
        if k == 1:
            d[0] = (sigma - 0.5) * log_p[0]
        return d, np.array([]), log_p

    # Edge weights: |log(p_{i+1}) - log(p_i)| for consecutive primes
    if weight_by_log_gap:
        weights = np.abs(np.diff(log_p))
    else:
        weights = np.ones(k - 1)

    # Main diagonal = weighted degree of path graph Laplacian
    d = np.zeros(k)
    d[0] = weights[0]
    d[k - 1] = weights[k - 2]
    for i in range(1, k - 1):
        d[i] = weights[i - 1] + weights[i]

    # Add potential: V_sigma = (sigma - 1/2) * diag(log p_i)
    delta = sigma - 0.5
    if abs(delta) > 0:
        d = d + delta * log_p

    # Sub-diagonal = negative edge weights (symmetric tridiagonal)
    e = -weights

    return d, e, log_p

# ---------------------------------------------------------------------------
# Complex-s extension (P4): Non-Hermitian operator for s in C
# ---------------------------------------------------------------------------

def default_prime_potential_complex(label: int, s: complex = 0.5 + 0j) -> complex:
    r"""Complex structural potential on a prime label.

    Extends :func:`default_prime_potential` to complex *s*:

        V(p, s) = (s - 1/2) \log p

    When s = 1/2 the potential vanishes (Hermitian / pure Laplacian).
    When s = 1/2 + it, the potential is purely imaginary: i t log(p),
    encoding oscillatory dynamics aligned with the Riemann zeros.

    TNFR physics basis: complex structural frequencies nu_f support
    oscillatory regimes in the nodal equation dEPI/dt = nu_f * DELTA_NFR.
    """
    return complex(s - 0.5) * np.log(float(label))

def build_h_tnfr_complex(
    G: nx.Graph,
    *,
    s: complex = 0.5 + 0j,
    potential_fn: Callable[[int, complex], complex] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Construct H_TNFR(s) for complex *s*, producing a non-Hermitian operator.

    .. math::

        H^{(k)}(s) = L_k + (s - 1/2)\,\mathrm{diag}(\log p_1, \ldots, \log p_k)

    When Im(s) != 0 the operator is non-Hermitian and its eigenvalues
    are complex.  This is the P4 extension of the TNFR-Riemann program.

    Parameters
    ----------
    G : nx.Graph
        Undirected prime-labeled graph.
    s : complex
        Complex structural parameter.  Re(s) = sigma, Im(s) = t.
    potential_fn : callable, optional
        Custom potential V(label, s) -> complex.  Defaults to
        :func:`default_prime_potential_complex`.

    Returns
    -------
    H, diag_V : tuple of ndarray
        H is a complex (n, n) matrix; diag_V is the diagonal potential.
    """
    if potential_fn is None:
        potential_fn = default_prime_potential_complex

    if not isinstance(G, nx.Graph):
        raise TypeError("G must be an undirected networkx.Graph")

    nodes = sorted(G.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    # Build weighted Laplacian (real, promoted to complex dtype)
    L: np.ndarray = np.zeros((n, n), dtype=complex)
    for u, v, data in G.edges(data=True):
        i = index[u]
        j = index[v]
        w = float(data.get("weight", 1.0))
        L[i, j] -= w
        L[j, i] -= w
        L[i, i] += w
        L[j, j] += w

    # Build complex diagonal potential
    diag_V: np.ndarray = np.zeros((n, n), dtype=complex)
    for node in nodes:
        i = index[node]
        label = G.nodes[node].get("label")
        if label is None:
            raise TNFRValueError(
                "All nodes must have an integer 'label' attribute",
                context={"node": node},
                suggestion="Assign integer labels to all nodes.",
            )
        diag_V[i, i] = potential_fn(int(label), s)

    H = L + diag_V
    return H, diag_V

def build_tridiagonal_h_tnfr_complex(
    count: int,
    s: complex = 0.5 + 0j,
    *,
    weight_by_log_gap: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Build tridiagonal H_TNFR(s) for complex *s* on a prime path graph.

    Like :func:`build_tridiagonal_h_tnfr` but with a complex main
    diagonal when Im(s) != 0.  The sub-diagonal remains real (from the
    Laplacian), so the matrix is *complex symmetric* but not Hermitian.

    Returns
    -------
    (d, e, log_primes)
        d : complex main diagonal, shape (count,).
        e : real sub-diagonal, shape (count - 1,).
        log_primes : real log(p_i) vector, shape (count,).
    """
    primes = _first_primes(count)
    log_p = np.array([np.log(float(p)) for p in primes])
    k = len(primes)

    if k < 2:
        d = np.zeros(k, dtype=complex)
        if k == 1:
            d[0] = (s - 0.5) * log_p[0]
        return d, np.array([]), log_p

    if weight_by_log_gap:
        weights = np.abs(np.diff(log_p))
    else:
        weights = np.ones(k - 1)

    # Main diagonal: Laplacian degree + complex potential
    d = np.zeros(k, dtype=complex)
    d[0] = weights[0]
    d[k - 1] = weights[k - 2]
    for i in range(1, k - 1):
        d[i] = weights[i - 1] + weights[i]

    delta_s = s - 0.5
    if abs(delta_s) > 0:
        d = d + delta_s * log_p

    e = -weights  # real sub-diagonal

    return d, e, log_p
