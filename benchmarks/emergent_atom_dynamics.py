"""Finite diffusion, graph-wave and spectral-coupling controls on chosen gaskets.

M1 evaluates exp(-t L_rw) with unit capacity on a directly constructed graph.
Its late-time fit and monotonic diagnostic check apply to the supplied field.
M2 separately supplies u_tt=-L_sym u. Frequency differences obey an algebraic
combination identity; neither radiation nor measured atomic lines are modeled.
M3 adds explicit graph edges and compares spectral gaps, without deriving a
chemical bond. M4 compares a whole-graph scalar diagnostic with one induced
child graph; it duplicates that child's value in a selected 0.4 inequality.
This is not a regional boundary budget, a U5 theorem or autonomous atom formation.
The local coherence_C function uses inverse mean magnitudes, not the mean of
canonical per-node coherence. Capacity and wave angular frequency are distinct.
No THOL operator is executed by the gasket constructor.

Status: auxiliary or finite evidence. See theory/EMERGENT_ONTOLOGY.md and
theory/NODAL_PARAMETER_FOUNDATIONS.md for model and physical-bridge limits.
"""

from __future__ import annotations

import pathlib
import sys

import networkx as nx
import numpy as np
from scipy.linalg import expm

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def sierpinski_simplex(m, levels):
    """Construct a corner-glued K_m gasket directly; no THOL execution."""
    if levels == 0:
        return nx.complete_graph(m), list(range(m))
    sub, subc = sierpinski_simplex(m, levels - 1)
    G = nx.Graph()
    copies = []
    for i in range(m):
        mp = {v: (i, v) for v in sub.nodes}
        G.add_nodes_from(mp[v] for v in sub.nodes)
        G.add_edges_from((mp[u], mp[v]) for u, v in sub.edges)
        copies.append([mp[c] for c in subc])
    parent = {n: n for n in G.nodes}

    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    for i in range(m):
        for j in range(i + 1, m):
            a, b = find(copies[i][j]), find(copies[j][i])
            if a != b:
                parent[b] = a
    H = nx.Graph()
    for u, v in G.edges:
        ru, rv = find(u), find(v)
        if ru != rv:
            H.add_edge(ru, rv)
    return H, [find(copies[i][i]) for i in range(m)]


def lrw_matrix(G):
    """Canonical EPI-channel operator L_rw (dNFR_epi = -L_rw * EPI)."""
    from tnfr.physics.structural_diffusion import (
        structural_diffusion_operator,
    )

    nodes, L = structural_diffusion_operator(G)
    return list(nodes), np.asarray(L, dtype=float)


def lsym_eigh(G):
    """Symmetric structural Laplacian L_sym spectrum (ascending) + vecs."""
    nodes = list(G.nodes)
    A = nx.to_numpy_array(G, nodelist=nodes)
    d = A.sum(axis=1)
    dinv = 1.0 / np.sqrt(d)
    L = np.eye(len(nodes)) - (dinv[:, None] * A * dinv[None, :])
    w, V = np.linalg.eigh(L)
    return np.clip(w, 0.0, None), V


def coherence_C(L, epi, nu_f=1.0):
    """Inverse-mean-magnitude EPI diagnostic for this declared diffusion model."""
    dnfr = -(L @ epi)
    depi = nu_f * dnfr
    return 1.0 / (1.0 + float(np.mean(np.abs(dnfr))) + float(np.mean(np.abs(depi))))


def main() -> None:
    print("=" * 70)
    print("PREPARED GASKET DIFFUSION AND AUXILIARY GRAPH-WAVE CONTROLS")
    print("=" * 70)
    rng = np.random.default_rng(0)

    # Prepared K4 gasket; no native THOL or physical atom construction
    G, _ = sierpinski_simplex(4, 2)
    nodes, L = lrw_matrix(G)
    N = len(nodes)
    w_sym, _ = lsym_eigh(G)
    lam2 = float(w_sym[w_sym > 1e-9][0])
    print(f"\nprepared K_4 gasket (grade 3), N={N}, " f"gap lambda_2={lam2:.4f}")

    # M1 -- DYNAMICS + C(t): excite, evolve, coherence rises + emits lambda_2
    print("\nM1 -- nodal DYNAMICS + C(t): dynamics makes structure observable")
    epi0 = rng.standard_normal(N)
    epi0 -= epi0.mean()
    T = 6.0 / lam2
    ts = np.linspace(0.0, T, 40)
    C_series, dnfr_series = [], []
    for t in ts:
        epi = expm(-t * L) @ epi0
        C_series.append(coherence_C(L, epi))
        dnfr_series.append(float(np.mean(np.abs(L @ epi))))
    C_series = np.array(C_series)
    dnfr_series = np.array(dnfr_series)
    mono = bool(np.all(np.diff(C_series) >= -1e-9))
    late = ts > T / 2
    rate = -np.polyfit(ts[late], np.log(dnfr_series[late]), 1)[0]
    ok = "OK" if abs(rate - lam2) < 0.02 else "OFF"
    print(
        f"  C: {C_series[0]:.3f} (excited) -> {C_series[-1]:.4f} "
        f"(relaxed), monotone={mono}"
    )
    print(
        f"  relaxation decay rate={rate:.4f} == "
        f"structural lambda_2={lam2:.4f} [{ok}]"
    )
    assert mono and C_series[-1] > 0.99
    assert abs(rate - lam2) < 0.02

    # M2 -- Auxiliary graph wave and algebraic frequency differences
    print("\nM2 -- AUXILIARY WAVE: omega_k=sqrt(lambda_k), frequency differences")
    w_sym2, V = lsym_eigh(G)
    omega = np.sqrt(w_sym2)
    terms = np.array(sorted({round(float(x), 6) for x in omega if x > 1e-6}))
    u0 = V[:, 1].copy()
    energy = []
    for t in np.linspace(0.0, 20.0, 200):
        u = V @ (np.cos(omega * t) * (V.T @ u0))
        ut = V @ (-(omega * np.sin(omega * t)) * (V.T @ u0))
        e_t = 0.5 * float(ut @ ut + u @ (V @ (w_sym2 * (V.T @ u))))
        energy.append(e_t)
    energy = np.array(energy)
    e_cons = float(np.std(energy) / (np.mean(energy) + 1e-12))
    lines = sorted(
        {round(float(a - b), 6) for a in terms for b in terms if a - b > 1e-6}
    )
    a, b, c = terms[-1], terms[-2], terms[-3]
    ritz = abs((a - c) - ((a - b) + (b - c))) < 1e-9
    print(
        f"  wave face OSCILLATES: energy std/mean={e_cons:.2e} "
        f"(conserved, vs M1 decay)"
    )
    print(f"  terms (sqrt lambda_k), first 4 = {np.round(terms[:4], 3)}")
    print(
        f"  algebraic positive frequency differences, count={len(lines)}; "
        f"Ritz holds={ritz}"
    )
    assert e_cons < 1e-6 and ritz

    # M3 -- Supplied cross-edges split the disconnected zero eigenspace
    print("\nM3 -- COUPLING/BONDING: two forms -> bonding/antibonding")
    atom, _ = sierpinski_simplex(3, 2)
    n_atom = atom.number_of_nodes()
    splits = []
    for nbonds in (1, 2, 4):
        mol = nx.disjoint_union(atom, atom)
        a_nodes = list(range(n_atom))
        b_nodes = list(range(n_atom, 2 * n_atom))
        for k in range(nbonds):
            mol.add_edge(a_nodes[k], b_nodes[k])
        w_mol, _ = lsym_eigh(mol)
        splits.append(float(w_mol[w_mol > 1e-9][0]))
    grows = splits[0] < splits[1] < splits[2]
    print(
        f"  antibonding split vs coupling (1,2,4 bonds) = "
        f"{[round(s, 4) for s in splits]}  grows={grows}"
    )
    print("  bonding mode stays at 0 (joint ground); bond = observable")
    assert grows

    # M4 -- Whole-graph and induced-child diagnostic comparison
    print("\nM4 -- FRACTAL coherence: C at parent + child scales")
    mol = nx.disjoint_union(atom, atom)
    a_nodes = list(range(n_atom))
    b_nodes = list(range(n_atom, 2 * n_atom))
    mol.add_edge(a_nodes[0], b_nodes[0])
    nodes_m, L_mol = lrw_matrix(mol)
    idx = {nd: i for i, nd in enumerate(nodes_m)}
    _, L_a = lrw_matrix(atom)
    epi_m = rng.standard_normal(mol.number_of_nodes())
    epi_m -= epi_m.mean()
    C_par = coherence_C(L_mol, epi_m)
    epi_child = np.array([epi_m[idx[nd]] for nd in a_nodes])
    C_child = coherence_C(L_a, epi_child - epi_child.mean())
    w_mol, _ = lsym_eigh(mol)
    lam2_mol = float(w_mol[w_mol > 1e-9][0])
    epi_relaxed = expm(-(8.0 / lam2_mol) * L_mol) @ epi_m
    C_par_relaxed = coherence_C(L_mol, epi_relaxed)
    u5 = C_par >= 0.4 * (C_child + C_child)  # alpha=0.4, illustrative
    print(f"  excited: C_parent(mol)={C_par:.3f}, " f"C_child(atom)={C_child:.3f}")
    print(f"  relaxed: C_parent={C_par_relaxed:.4f} -> resonant attractor")
    print(f"  U5 multi-scale C_parent >= alpha*sum C_child holds={bool(u5)}")
    assert C_par_relaxed > 0.99 and u5

    print("\n" + "=" * 70)
    print("RESULT SCOPE: one prepared gasket supports the recorded diffusion checks.")
    print("The late-time rate is a fitted comparison with its spectral gap.")
    print("A separate graph-wave equation supplies the oscillations in M2.")
    print(
        "Its frequency differences satisfy a telescoping identity, not an emission law."
    )
    print("Explicitly added graph edges change the selected gaps in M3.")
    print(
        "M4 uses a chosen coherence proxy and an induced child without boundary exchange."
    )
    print("Its duplicated-child inequality is a finite check, not a U5 certificate.")
    print("THOL birth, autonomous maintenance and radiation are not implemented here.")
    print("Atomic spectra and molecular bonds have not been physically identified.")
    print("All graph sizes, initial fields and auxiliary model choices remain inputs.")
    print("=" * 70)


if __name__ == "__main__":
    main()
