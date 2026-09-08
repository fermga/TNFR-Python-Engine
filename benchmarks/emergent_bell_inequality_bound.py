"""CHSH diagnostic for two explicitly Bell-local shared-angle models.

This benchmark answers a narrow question: what CHSH value is produced when
both measurement outcomes are deterministic functions of a shared classical
angle and of the local analyzer setting? It evaluates two angle distributions:

* M1 samples a uniform angle.
* M2 obtains the angle from an auxiliary Kuramoto simulation on a
  self-similar graph.

For both cases the response functions have the Bell-local form

    A = A(x, lambda),    B = B(y, lambda),

with no dependence of A on y or B on x. The CHSH bound |S| <= 2 therefore holds
by construction for every distribution of lambda. The code also checks the
single-trial algebraic bound before averaging. M2 shows that replacing the
uniform distribution with this particular self-similar, resonant generator
does not change that fact. It does not test every TNFR measurement model.

The graph used by M2 is a Sierpinski-style support built directly in this file.
It is not produced by executing the THOL operator. Its Kuramoto update is an
auxiliary synchronous nearest-neighbour phase model, not an invocation of the
TNFR nodal engine. The measured order parameter documents synchronization in
that declared model.

Local graph updates and Bell locality are different statements. Bell locality
is a conditional factorization property of outcomes in a spacetime
measurement protocol, together with assumptions such as measurement-setting
independence. Nearest-neighbour terms in an evolution equation do not prove
that factorization. They also do not establish a strict relativistic causal
cone. For a finite continuous-time graph wave,

    cos(t sqrt(L)) = I - t^2 L / 2! + t^4 L^2 / 4! - ...,

so a matrix entry at graph distance d can generally appear at order t^(2d);
nonzero analytic tails may occur at every distance for any t > 0. A discrete
synchronous stencil has a finite dependency radius per step, but that is a
property of the numerical schedule. Neither observation proves Lorentz
invariance. That would require a specified spacetime limit and the relevant
transformation symmetry, not only a low-wave-number dispersion fit.

The result also does not decide whether TNFR can support a Born-type
measurement rule or single-particle interference. Bell inequalities constrain
bipartite correlations under their stated assumptions; they are not a no-go
theorem for the Born rule as a whole or for single-particle interference. This
script defines no probability amplitude, detector model, or TNFR measurement
map. The per-node Poincare-sphere field in the auxiliary substrate is
classical polarization, but this benchmark merely uses an analogous shared
angle and does not extract it from polarization_density.

Likewise, the absence of a calibrated action variable in the current Hz_str
model means that this benchmark cannot derive a physical value of hbar. It
does not prove that a future, empirically anchored bridge is impossible.

Measured claims:

* each explicit deterministic response model satisfies |S| <= 2;
* at the selected angles, every sampled single-trial CHSH value is -2,
  so both sampled distributions saturate the bound;
* the auxiliary M2 generator reaches the reported Kuramoto order parameter.

Status: RESEARCH NEGATIVE CONTROL. It validates the constructed local
hidden-variable response family and no broader causal or quantum no-go claim.

Run:
    python benchmarks/emergent_bell_inequality_bound.py
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np


def _sierpinski_simplex(m: int, levels: int) -> nx.Graph:
    """Build the self-similar graph used by the auxiliary M2 generator."""
    if levels == 0:
        return nx.complete_graph(m)
    sub = _sierpinski_simplex(m, levels - 1)
    sub_corners = list(range(m)) if levels == 1 else list(sub.nodes())[:m]
    graph = nx.Graph()
    copies: list[list[tuple]] = []
    for i in range(m):
        mapping = {v: (i, v) for v in sub.nodes}
        graph.add_nodes_from(mapping[v] for v in sub.nodes)
        graph.add_edges_from((mapping[u], mapping[v]) for u, v in sub.edges)
        copies.append([mapping[c] for c in sub_corners])
    parent = {node: node for node in graph.nodes}

    def find(node):
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    for i in range(m):
        for j in range(i + 1, m):
            root_a = find(copies[i][j])
            root_b = find(copies[j][i])
            if root_a != root_b:
                parent[root_b] = root_a
    merged = nx.Graph()
    for u, v in graph.edges:
        root_u = find(u)
        root_v = find(v)
        if root_u != root_v:
            merged.add_edge(root_u, root_v)
    return merged


def fractal_resonant_hidden_variable(
    rng: np.random.Generator,
    *,
    m: int = 4,
    levels: int = 2,
    coupling: float = 2.0,
    steps: int = 150,
    dt: float = 0.05,
) -> tuple[float, float]:
    """Return the common phase of the auxiliary neighbour-coupled model.

    The support is self-similar and the Kuramoto update reads graph neighbours.
    This is an input generator for the explicit hidden-variable response map,
    not a proof of Bell locality or a canonical THOL execution. The return
    values are the circular mean phase and Kuramoto order parameter R.
    """
    graph = _sierpinski_simplex(m, levels)
    nodes = list(graph.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    theta = rng.uniform(0.0, 2.0 * math.pi, n)
    adjacency: list[list[int]] = [[] for _ in range(n)]
    for u, v in graph.edges():
        adjacency[index[u]].append(index[v])
        adjacency[index[v]].append(index[u])
    for _ in range(steps):
        updated = theta.copy()
        for i in range(n):
            neighbours = adjacency[i]
            if not neighbours:
                continue
            drive = sum(math.sin(theta[j] - theta[i]) for j in neighbours)
            updated[i] = theta[i] + dt * coupling * drive / len(neighbours)
        theta = updated
    order = complex(np.mean(np.exp(1j * theta)))
    return float(np.angle(order)), float(abs(order))


def _chsh_from_shared_angles(
    lam: np.ndarray,
    a: float,
    a_prime: float,
    b: float,
    b_prime: float,
) -> tuple[float, float, float, float, float, float]:
    """Evaluate the deterministic shared-angle model and its pointwise bound.

    Returns the four correlations, their CHSH combination, and the largest
    absolute single-trial CHSH combination. The response functions are
    dichotomic except at measure-zero threshold ties, where numpy.sign returns
    zero; either case obeys the pointwise bound.
    """
    angles = np.asarray(lam, dtype=float)
    if angles.ndim != 1 or angles.size == 0:
        raise ValueError("lam must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(angles)):
        raise ValueError("lam must contain only finite angles")

    def wing_a(setting: float) -> np.ndarray:
        return np.sign(np.cos(2.0 * (angles - setting))).astype(np.int8)

    def wing_b(setting: float) -> np.ndarray:
        shifted = angles + math.pi / 2.0 - setting
        return np.sign(np.cos(2.0 * shifted)).astype(np.int8)

    response_a = wing_a(a)
    response_ap = wing_a(a_prime)
    response_b = wing_b(b)
    response_bp = wing_b(b_prime)

    product_ab = response_a * response_b
    product_abp = response_a * response_bp
    product_apb = response_ap * response_b
    product_apbp = response_ap * response_bp

    e_ab = float(np.mean(product_ab))
    e_abp = float(np.mean(product_abp))
    e_apb = float(np.mean(product_apb))
    e_apbp = float(np.mean(product_apbp))
    s = e_ab - e_abp + e_apb + e_apbp

    pointwise = product_ab - product_abp + product_apb + product_apbp
    max_pointwise = float(np.max(np.abs(pointwise)))
    return e_ab, e_abp, e_apb, e_apbp, s, max_pointwise


def chsh_local_hidden_variable(
    a: float,
    a_prime: float,
    b: float,
    b_prime: float,
    *,
    n: int = 2_000_000,
    seed: int = 0,
) -> tuple[float, float, float, float, float]:
    """Return CHSH correlations for the uniform shared-angle LHV model.

    The public five-value return shape is retained for compatibility. The
    response factorization, rather than the origin of the shared angle, is
    what makes this particular model Bell-local.
    """
    rng = np.random.default_rng(seed)
    lam = rng.uniform(0.0, 2.0 * math.pi, n)
    return _chsh_from_shared_angles(
        lam, a, a_prime, b, b_prime
    )[:5]


def quantum_singlet_chsh(
    a: float,
    a_prime: float,
    b: float,
    b_prime: float,
) -> float:
    """Return the textbook singlet CHSH prediction for comparison only."""

    def correlation(x: float, y: float) -> float:
        return -math.cos(2.0 * (x - y))

    return (
        correlation(a, b)
        - correlation(a, b_prime)
        + correlation(a_prime, b)
        + correlation(a_prime, b_prime)
    )


def main() -> None:
    print("=" * 74)
    print("CHSH DIAGNOSTIC FOR EXPLICIT SHARED-ANGLE RESPONSE MODELS")
    print("=" * 74)

    a = 0.0
    a_prime = math.pi / 4.0
    b = math.pi / 8.0
    b_prime = 3.0 * math.pi / 8.0

    print("\n[M1] Uniform shared-angle local hidden-variable control.")
    rng = np.random.default_rng(0)
    lam = rng.uniform(0.0, 2.0 * math.pi, 2_000_000)
    e_ab, e_abp, e_apb, e_apbp, s, pointwise_max = (
        _chsh_from_shared_angles(lam, a, a_prime, b, b_prime)
    )
    print(f"     E(a,b)   = {e_ab:+.4f}")
    print(f"     E(a,b')  = {e_abp:+.4f}")
    print(f"     E(a',b)  = {e_apb:+.4f}")
    print(f"     E(a',b') = {e_apbp:+.4f}")
    print(f"     CHSH  S  = {s:+.4f}")
    print(f"     maximum pointwise |CHSH| = {pointwise_max:.4f}")
    print("     local hidden-variable bound: |S| <= 2.0000")
    s_q = quantum_singlet_chsh(a, a_prime, b, b_prime)
    print(f"     Tsirelson bound for reference: {2 * math.sqrt(2):.4f}")
    print(f"     singlet prediction at these angles: {s_q:+.4f}")
    assert pointwise_max <= 2.0
    assert abs(s) <= 2.0 + 1e-12
    assert abs(abs(s) - 2.0) < 0.01
    print("     -> PASS: the explicitly factorized response saturates its")
    print("        classical bound at these settings.")

    print("\n[M2] Auxiliary self-similar, neighbour-coupled angle generator.")
    rng = np.random.default_rng(0)
    n_trials = 4000
    lam = np.empty(n_trials)
    order_r = np.empty(n_trials)
    for trial in range(n_trials):
        lam[trial], order_r[trial] = fractal_resonant_hidden_variable(rng)
    print(
        f"     mean Kuramoto order parameter R = {order_r.mean():.4f} "
        "(auxiliary synchrony)"
    )
    e_ab2, e_abp2, e_apb2, e_apbp2, s2, pointwise_max2 = (
        _chsh_from_shared_angles(lam, a, a_prime, b, b_prime)
    )
    print(f"     E(a,b)   = {e_ab2:+.4f}")
    print(f"     E(a,b')  = {e_abp2:+.4f}")
    print(f"     E(a',b)  = {e_apb2:+.4f}")
    print(f"     E(a',b') = {e_apbp2:+.4f}")
    print(f"     CHSH  S  = {s2:+.4f}")
    print(f"     maximum pointwise |CHSH| = {pointwise_max2:.4f}")
    assert pointwise_max2 <= 2.0
    assert abs(s2) <= 2.0 + 1e-12
    assert abs(abs(s2) - 2.0) < 0.05
    print("     -> PASS: changing the shared-angle generator does not alter")
    print("        the bound while the response factorization is unchanged.")

    print("\n" + "=" * 74)
    print("RESULT: both constructed response models satisfy CHSH because their")
    print("outcomes are Bell-factorized by definition. This does not derive a")
    print("TNFR causal cone, Lorentz invariance, or a general TNFR measurement")
    print("theory, and it does not settle the Born rule, single-particle")
    print("interference, or a physical value of hbar.")
    print("=" * 74)


if __name__ == "__main__":
    main()
