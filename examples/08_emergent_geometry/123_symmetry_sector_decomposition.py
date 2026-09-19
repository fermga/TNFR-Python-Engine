#!/usr/bin/env python3
"""Example 123 — finite automorphism-sector checks on selected graphs.

For each graph in a fixed test family, the script enumerates its automorphisms,
builds their permutation matrices and averages them to obtain the invariant
projector ``P_triv``. It then checks, numerically, that the random-walk
diffusion matrix commutes with those permutations, that the projector rank
equals the number of vertex orbits, and that the invariant subspace is
preserved.

Two additional comparisons have narrower scope. First, a uniformly seeded graph
is passed to ``extract_phase_space_point`` and its resulting ``Phi_s`` vector is
projected. This auxiliary model is initialized from the supplied graph state;
the engine dynamics do not derive it. Second,
eigenvectors returned for the vertex-transitive complete and cycle graphs are
projected onto the invariant subspace. Those finite checks do not imply that all
node information outside an orbit is spectral, that every possible read-out lies
in one of the observed locations, or that degree resolves every orbit.

The decomposition ``V = Fix(G) (+) Fix(G)^perp`` is ordinary finite
representation theory for the explicitly defined graph action. No map from
``S(T)`` or any Millennium-problem target into these spaces is constructed, so
the example makes no Riemann-obstruction identification.

Status: RESEARCH example; finite graph algebra with scoped auxiliary read-outs.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import networkx as nx
import numpy as np
from networkx.algorithms.isomorphism import GraphMatcher

from tnfr.alias import set_attr
from tnfr.constants.aliases import ALIAS_EPI, ALIAS_VF
from tnfr.dynamics import default_compute_delta_nfr
from tnfr.physics.structural_diffusion import structural_diffusion_operator
from tnfr.physics.symplectic_substrate import extract_phase_space_point


def _perm_matrix(mapping, nodes):
    idx = {nd: i for i, nd in enumerate(nodes)}
    n = len(nodes)
    P = np.zeros((n, n))
    for src, dst in mapping.items():
        P[idx[dst], idx[src]] = 1.0
    return P


def _automorphisms(G, cap=2000):
    out = []
    for m in GraphMatcher(G, G).isomorphisms_iter():
        out.append(m)
        if len(out) >= cap:
            break
    return out


def _orbit_count(auts, nodes):
    """Number of vertex orbits via union-find over the automorphisms."""
    idx = {nd: i for i, nd in enumerate(nodes)}
    parent = list(range(len(nodes)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for m in auts:
        for s, d in m.items():
            ra, rb = find(idx[s]), find(idx[d])
            if ra != rb:
                parent[ra] = rb
    return len({find(i) for i in range(len(nodes))})


def _trivial_projector(auts, nodes):
    n = len(nodes)
    P = np.zeros((n, n))
    for m in auts:
        P += _perm_matrix(m, nodes)
    return P / len(auts)


def _seed_symmetric(G):
    for nd in G.nodes():
        G.nodes[nd]["theta"] = 0.3
        set_attr(G.nodes[nd], ALIAS_EPI, 0.2)
        set_attr(G.nodes[nd], ALIAS_VF, 1.0)
    default_compute_delta_nfr(G)


def _test_graphs():
    return [
        ("cycle C8 (D8)", nx.cycle_graph(8)),
        ("complete K6 (S6)", nx.complete_graph(6)),
        ("star K1,5 (S5)", nx.star_graph(5)),
        ("path P6 (Z2)", nx.path_graph(6)),
        ("torus C3xC3", nx.cartesian_product(nx.cycle_graph(3), nx.cycle_graph(3))),
    ]


def experiment_1_equivariance_orbits():
    """M1-M3: equivariance, rank(P_triv)=#orbits, L_rw preserves Fix(G)."""
    print("=" * 74)
    print("EXPERIMENT 1: Equivariance, dim Fix(G) = #orbits, L_rw preserves it")
    print("=" * 74)
    print("For each selected graph, enumerate automorphisms and test whether L_rw")
    print("commutes with them. Compare projector rank with the orbit count.")
    print()
    print(
        f"  {'graph':22s} {'|Aut|':>6} {'n':>3} {'orbits':>7} "
        f"{'rank':>5} {'equiv':>8} {'preserve':>9}"
    )
    out = {}
    for name, G in _test_graphs():
        nodes, L = structural_diffusion_operator(G)
        auts = _automorphisms(G)
        max_comm = max(
            float(
                np.linalg.norm(_perm_matrix(m, nodes) @ L - L @ _perm_matrix(m, nodes))
            )
            for m in auts
        )
        P_triv = _trivial_projector(auts, nodes)
        rank = int(np.linalg.matrix_rank(P_triv, tol=1e-9))
        orbits = _orbit_count(auts, nodes)
        pres = float(np.linalg.norm(L @ P_triv - P_triv @ L))
        out[name] = (nodes, L, P_triv, orbits)
        print(
            f"  {name:22s} {len(auts):>6} {len(nodes):>3} {orbits:>7} "
            f"{rank:>5} {max_comm:>8.1e} {pres:>9.1e}"
        )
    print()
    print("  -> On this finite family: commutators vanish, rank=orbits, and")
    print("     the invariant subspace and its orthogonal complement are preserved.")
    return out


def experiment_2_symmetric_auxiliary_projection(results):
    """Project an auxiliary Phi_s read-out obtained from a symmetric seed."""
    print()
    print("=" * 74)
    print("EXPERIMENT 2: Auxiliary Phi_s Read-out from a Symmetric Seed")
    print("=" * 74)
    print("Initialize the auxiliary phase-space model from a uniform graph state")
    print("and measure whether its Phi_s vector is invariant under P_triv.")
    print()
    print(
        f"  {'graph':22s} {'orbits':>7} {'||v-P_triv v||':>15} " f"{'sigma(Phi_s)':>13}"
    )
    for name, G in _test_graphs():
        nodes, L, P_triv, orbits = results[name]
        _seed_symmetric(G)
        p = extract_phase_space_point(G)
        v = np.array([p.phi_s[i] for i in range(len(nodes))], dtype=float)
        resid = float(np.linalg.norm(v - P_triv @ v))
        print(f"  {name:22s} {orbits:>7} {resid:>15.1e} {np.std(v):>13.1e}")
    print()
    print("  -> The selected symmetric initialization gives an invariant Phi_s")
    print("     vector. This is not a claim about arbitrary states or trajectories.")


def experiment_3_degree_orbit_comparison():
    """Show that degree is orbit-invariant and can be coarser than the orbits."""
    print()
    print("=" * 74)
    print("EXPERIMENT 3: Degree Is Orbit-Invariant but Need Not Resolve Orbits")
    print("=" * 74)
    print("Degree is constant within every automorphism orbit, but distinct orbits")
    print("can share a degree. It is one graph statistic, not the full substrate.")
    print()
    for name, G in [
        ("star K1,5 (S5)", nx.star_graph(5)),
        ("path P6 (Z2)", nx.path_graph(6)),
        ("complete K6 (S6)", nx.complete_graph(6)),
    ]:
        degs = sorted({d for _, d in G.degree()})
        print(
            f"  {name:22s} distinct per-node degrees = {degs} "
            f"({len(degs)} class(es))"
        )
    print()
    print("  -> star: two degree classes; path: three orbits collapse to two")
    print("     degree values; complete: one class. The calculation explicitly")
    print("     refutes the claim that this statistic always resolves all orbits.")


def experiment_4_transitive_eigenvector_projections(results):
    """Project eigenvectors for two selected vertex-transitive graphs."""
    print()
    print("=" * 74)
    print("EXPERIMENT 4: Eigenvector Projections on Two Transitive Graphs")
    print("=" * 74)
    print("Project the numerically returned L_rw eigenvectors onto the invariant")
    print("subspace for K6 and C8. Degenerate eigenspace bases are solver-chosen.")
    print()
    for name in ["complete K6 (S6)", "cycle C8 (D8)"]:
        nodes, L, P_triv, orbits = results[name]
        w, V = np.linalg.eig(L)
        order = np.argsort(w.real)
        fracs = []
        for k in range(len(nodes)):
            vk = V[:, order[k]].real
            vk = vk / (np.linalg.norm(vk) + 1e-15)
            fracs.append(float(np.linalg.norm(P_triv @ vk)))
        print(f"  {name}: ||P_triv v_k|| by eigenvalue:")
        print("    " + " ".join(f"{t:.2f}" for t in fracs))
    print()
    print("  -> In these two transitive examples, the constant mode projects onto")
    print("     Fix(G) and the reported nonconstant modes project onto its complement.")
    print("     No location for arithmetic or S(T) is inferred.")


# Historical callable names remain aliases for compatibility only.
experiment_2_substrate_in_fix = experiment_2_symmetric_auxiliary_projection
experiment_3_orbit_resolution = experiment_3_degree_orbit_comparison
experiment_4_discriminating_spectrum = experiment_4_transitive_eigenvector_projections


def main():
    print()
    print("  TNFR Example 123: Finite Automorphism-Sector Checks")
    print("  ===================================================")
    print()
    results = experiment_1_equivariance_orbits()
    experiment_2_symmetric_auxiliary_projection(results)
    experiment_3_degree_orbit_comparison()
    experiment_4_transitive_eigenvector_projections(results)
    print()
    print("=" * 74)
    print("SCOPED RESULT")
    print("=" * 74)
    print("For the selected finite graphs, L_rw commutes with the enumerated")
    print("automorphisms and preserves their invariant subspaces. A symmetric")
    print("initialization yields an invariant auxiliary Phi_s vector, and the")
    print("reported eigenvectors on two transitive graphs have the shown")
    print("projections. These checks do not locate all information, characterize")
    print("engine reachability, generate a symplectic substrate, or define an")
    print("S(T) map or Riemann obstruction.")


if __name__ == "__main__":
    main()
