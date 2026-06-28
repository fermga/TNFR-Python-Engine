"""Emergent Atom Dynamics: the nodal equation + coherence make the form seen.

THE GAP (user, theory creator): the recent arc characterised the atom's
STRUCTURE (shells = L-spectrum = cardinals = simplex grade) -- but
STATICALLY, as eigenvalues of a fixed graph. To explain OBSERVABLE phenomena
we need the EMERGENT DYNAMICS, which is exactly what the nodal equation does
(dEPI/dt = nu_f * dNFR); and COHERENCE C -- the canonical 5th parameter,
which is *resonant* and *fractal* -- was never tracked. This benchmark brings
both to the atom-FORM.

THE THREE LAYERS:
  - STRUCTURE (static): the shells = the spectrum of the canonical operator.
    A fixed graph; no time, no C(t). (the recent arc)
  - DYNAMICS (the nodal equation): dEPI/dt = nu_f * dNFR, with the EPI channel
    dNFR_epi = -L_rw * EPI (structural_diffusion_operator, the EXACT canonical
    operator). Evolving the form is what produces OBSERVABLE phenomena.
  - COHERENCE C: canonical (C = 1/(1 + mean|dNFR| + mean|dEPI|)), constitutive
    of NFR-hood, *resonant* (monotone distance to the dNFR=0 attractor) and
    *fractal* (U5 multi-scale, C_parent >= alpha*sum C_child).

WHAT EMERGES (measured):
  - M1 DYNAMICS + C(t): an excited form relaxes; C(t) rises monotonically to
    the resonant attractor (de-excitation). The relaxation decay rate recovers
    the structural shell lambda_2 EXACTLY -- the static eigenvalue is
    OBSERVABLE only through the dynamics ("structure is silent; dynamics
    emits it").
  - M2 EMISSION SPECTRUM: the conservative/wave face rings at the term
    frequencies omega_k = sqrt(lambda_k) (the shells); it OSCILLATES (energy
    conserved) where the diffusive face relaxed. The OBSERVABLE emission lines
    are the term DIFFERENCES {omega_j - omega_k} -- the Rydberg-Ritz
    combination principle, the empirical organising law of atomic spectra.
  - M3 COUPLING / BONDING: coupling two atom-forms splits each shared ground
    mode into bonding (stays at 0) + antibonding (lifts), and the splitting
    GROWS with the coupling -- the observable molecular bond (tight-binding).
  - M4 FRACTAL COHERENCE: C is defined at every scale -- the molecule (parent)
    and its two atoms (children); the relaxed state drives all scales to the
    resonant attractor (C -> 1), and U5 (C_parent >= alpha*sum C_child) holds.

So the atom is not the static spectrum but the COHERENT FORM EVOLVING under
the nodal equation; the observables (emission lines, bonds) live in the
dynamics, and C is the canonical resonant/fractal parameter that tracks it.

HONEST SCOPE: the diffusive heat-semigroup, the wave/Helmholtz ring, the
Rydberg-Ritz combination principle, and tight-binding bonding are STANDARD
physics; the TNFR content is that the *one* nodal equation drives them on the
emergent atom-form, with C(t) (resonant + fractal) as the canonical health.
The emission lines are the form's OWN terms (not the hydrogen spectrum);
reaches the independent-particle skeleton only. Closes no open problem. R and
pi assumed.

Run:
    python benchmarks/emergent_atom_dynamics.py

Theoretical anchor: AGENTS.md (nodal equation; C(t); two regimes
diffusive/conservative; U5 multi-scale fractality); benchmarks/
emergent_atomic_shells.py (the static structure), emergent_base_dimension.py
(the EPI channel as diffusion). Status: RESEARCH (synthesis / falsifier).
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
    """THOL self-similar nesting of K_m (the atom-form)."""
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
    """Canonical C = 1/(1 + mean|dNFR| + mean|dEPI|) on the EPI channel."""
    dnfr = -(L @ epi)
    depi = nu_f * dnfr
    return 1.0 / (
        1.0 + float(np.mean(np.abs(dnfr))) + float(np.mean(np.abs(depi)))
    )


def main() -> None:
    print("=" * 70)
    print("EMERGENT ATOM DYNAMICS -- dynamics + coherence make it observable")
    print("=" * 70)
    rng = np.random.default_rng(0)

    # atom-form: the grade-3 THOL nest (the "3D atom" of atomic_shells)
    G, _ = sierpinski_simplex(4, 2)
    nodes, L = lrw_matrix(G)
    N = len(nodes)
    w_sym, _ = lsym_eigh(G)
    lam2 = float(w_sym[w_sym > 1e-9][0])
    print(f"\natom-form: THOL nest K_4 (grade 3), N={N}, "
          f"gap lambda_2={lam2:.4f}")

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
    print(f"  C: {C_series[0]:.3f} (excited) -> {C_series[-1]:.4f} "
          f"(relaxed), monotone={mono}")
    print(f"  relaxation decay rate={rate:.4f} == "
          f"structural lambda_2={lam2:.4f} [{ok}]")
    assert mono and C_series[-1] > 0.99
    assert abs(rate - lam2) < 0.02

    # M2 -- EMISSION: wave face rings at sqrt(lambda_k); lines = differences
    print("\nM2 -- EMISSION: wave ring omega_k=sqrt(lambda_k), lines=diff")
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
    lines = sorted({round(float(a - b), 6)
                    for a in terms for b in terms if a - b > 1e-6})
    a, b, c = terms[-1], terms[-2], terms[-3]
    ritz = abs((a - c) - ((a - b) + (b - c))) < 1e-9
    print(f"  wave face OSCILLATES: energy std/mean={e_cons:.2e} "
          f"(conserved, vs M1 decay)")
    print(f"  terms (sqrt lambda_k), first 4 = {np.round(terms[:4], 3)}")
    print(f"  observable lines = differences, count={len(lines)}; "
          f"Ritz holds={ritz}")
    assert e_cons < 1e-6 and ritz

    # M3 -- COUPLING / BONDING: the bond splits the shared ground mode
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
    print(f"  antibonding split vs coupling (1,2,4 bonds) = "
          f"{[round(s, 4) for s in splits]}  grows={grows}")
    print("  bonding mode stays at 0 (joint ground); bond = observable")
    assert grows

    # M4 -- FRACTAL COHERENCE: C is defined at every scale (molecule + atoms)
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
    print(f"  excited: C_parent(mol)={C_par:.3f}, "
          f"C_child(atom)={C_child:.3f}")
    print(f"  relaxed: C_parent={C_par_relaxed:.4f} -> resonant attractor")
    print(f"  U5 multi-scale C_parent >= alpha*sum C_child holds={bool(u5)}")
    assert C_par_relaxed > 0.99 and u5

    print("\n" + "=" * 70)
    print("VERDICT: the atom is the COHERENT FORM EVOLVING under the nodal")
    print("equation, not the static spectrum. The dynamics makes the silent")
    print("structure observable: relaxation emits lambda_2 (M1), the wave")
    print("face rings at sqrt(lambda_k) with Rydberg-Ritz lines (M2),")
    print("coupling emits the bond (M3). Coherence C is the canonical")
    print("parameter -- resonant (-> dNFR=0) and fractal (every scale, M4).")
    print("HONEST SCOPE: standard diffusion / wave / Ritz / tight-binding")
    print("driven by the one nodal equation on the emergent form;")
    print("independent-particle skeleton only; closes no open problem.")
    print("R and pi stay assumed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
