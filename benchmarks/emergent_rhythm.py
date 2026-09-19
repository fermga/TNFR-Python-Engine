"""Auxiliary graph-wave oscillations, beats and hierarchical spectra.

On a supplied symmetric normalized Laplacian L, compare q_dot=-Lq with the
separately declared conservative model q_ddot=-Lq. The wave has the exact modal
frequencies sqrt(lambda_k) and energy (||q_dot||^2 + q.T L q)/2. Its initial
displacement and velocity (zero here), stiffness, and clock are model inputs.
For irregular graphs q is the degree-weighted symmetric coordinate, not the
raw scalar EPI coordinate. The cycle used for the evolution is regular.

The experiment retains useful finite observations: wave-energy conservation,
a difference-frequency component of a quadratic local detector, passages
below a selected relative pressure threshold, and spectral multiplicities of
a prescribed nested graph. A relative low-pressure passage need not approach
zero pressure, and even zero wave pressure with nonzero velocity is not a
stationary state. Beats are not generally equilibria or whole-state periods.
Repeated eigenvalues alone do not establish the same beat across scales.

The graph wave is distinct from the repository's isotropic auxiliary
symplectic substrate, whose stiffness is the identity. Neither model is
derived here as an autonomous completion of the nodal EPI law. No particles,
arithmetic correspondence, canonical THOL realization or TNFR-wide sustained
vibration is established by this benchmark.

Run:
    python benchmarks/emergent_rhythm.py

Related owners: src/tnfr/physics/structural_diffusion.py (shared Laplacian and
damped/undamped wave comparisons) and theory/TNFR_VARIATIONAL_PRINCIPLE.md
(diffusion, graph-wave and substrate scope). Status: auxiliary research.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.linalg import expm

from tnfr.physics.structural_diffusion import symmetric_normalized_laplacian


def lsym_eigh(G):
    """Diagonalize the shared Laplacian, retaining stationary isolate modes."""
    _, L = symmetric_normalized_laplacian(G)
    w, V = np.linalg.eigh(L)
    return np.clip(w, 0.0, None), V, L


def sierpinski_simplex(m, levels):
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


def main() -> None:
    print("=" * 70)
    print("AUXILIARY GRAPH WAVE -- oscillations and spectral beats")
    print("=" * 70)

    G = nx.cycle_graph(36)
    w, V, L = lsym_eigh(G)
    omega = np.sqrt(w)

    # M1 -- Two declared models on the same fixed symmetric geometry.
    print("\nM1 -- conservative graph wave and dissipative diffusion:")
    epi0 = V[:, 1] + 0.7 * V[:, 3] + 0.5 * V[:, 6]
    diss = [
        float(np.sum(np.abs(L @ (expm(-t * L) @ epi0)))) for t in (0.0, 5.0, 20.0, 60.0)
    ]
    ts = np.linspace(0.0, 120.0, 1200)
    energy, pressure = [], []
    for t in ts:
        e = V @ (np.cos(omega * t) * (V.T @ epi0))
        et = V @ (-(omega * np.sin(omega * t)) * (V.T @ epi0))
        energy.append(0.5 * float(et @ et + e @ (L @ e)))
        pressure.append(float(np.sum(np.abs(L @ e))))
    energy = np.array(energy)
    pressure = np.array(pressure)
    e_cons = float(np.std(energy) / np.mean(energy))
    print(
        f"  dissipative pressure decays {diss[0]:.3f} -> {diss[-1]:.3f}"
        f" (finite-horizon relaxation)"
    )
    print(
        f"  conservative energy conserved (std/mean={e_cons:.1e}); pressure"
        f" sustained mean={pressure.mean():.3f} std={pressure.std():.3f}"
    )
    assert diss[-1] < 0.3 * diss[0] and e_cons < 1e-6 and pressure.std() > 1e-3

    # M2 -- A detector can contain a modal difference frequency.
    print("\nM2 -- detector beat at omega_b - omega_a (two selected modes):")
    ka, kb = 4, 6
    oa, ob = omega[ka], omega[kb]
    beat = abs(ob - oa)
    epi2 = V[:, ka] + V[:, kb]
    tt = np.linspace(0.0, 40.0 / beat, 4000)
    node = int(np.argmax(np.abs(V[:, ka])))
    sig = np.array([(V[node, :] @ (np.cos(omega * t) * (V.T @ epi2))) ** 2 for t in tt])
    sd = sig - sig.mean()
    spec = np.abs(np.fft.rfft(sd))
    freqs = 2 * np.pi * np.fft.rfftfreq(len(sd), d=tt[1] - tt[0])
    # the BEAT is the slow envelope: dominant peak BELOW the lower resonance
    low = (freqs > 1e-6) & (freqs < oa)
    f_peak = float(freqs[low][np.argmax(spec[low])])
    print(f"  resonances omega_a={oa:.4f}, omega_b={ob:.4f}, beat=" f"{beat:.4f}")
    print(
        f"  quadratic detector dominant slow frequency={f_peak:.4f} "
        f"(== beat: {abs(f_peak - beat) < 0.08 * beat})"
    )
    assert abs(f_peak - beat) < 0.08 * beat

    # M3 -- The relative threshold does not test zero pressure or stationarity.
    print("\nM3 -- passages below a selected relative pressure threshold:")
    dn = np.array(
        [
            float(np.sum(np.abs(L @ (V @ (np.cos(omega * t) * (V.T @ epi2))))))
            for t in tt
        ]
    )
    lo = dn.min()
    hi = dn.max()
    thr = lo + 0.25 * (hi - lo)
    below = dn < thr
    passages = int(np.sum(below[1:] & ~below[:-1]))
    print(f"  structural pressure pulses {lo:.3f}..{hi:.3f}; the system")
    print(f"  crosses below threshold {thr:.3f} a total of {passages} times")
    print("  => detector passages; neither equilibrium nor periodicity certified")
    assert hi > 2 * (lo + 1e-9) and passages >= 3

    # M4 -- Read the spectrum of a separately constructed nested graph.
    print("\nM4 -- auxiliary wave spectrum and a supplied nested geometry:")
    print(
        f"  RESONANT: built from omega_k=sqrt(lambda_k), span "
        f"{omega[1]:.3f}..{omega[-1]:.3f} (graph modes)"
    )
    nest, _ = sierpinski_simplex(4, 3)
    wn, _, _ = lsym_eigh(nest)
    _, counts = np.unique(np.round(wn, 6), return_counts=True)
    max_mult = int(counts.max())
    print(f"  NESTED  : prescribed graph (N={nest.number_of_nodes()}) has a")
    print(f"            largest rounded spectral multiplicity of {max_mult}")
    print("            (multiplicity alone does not establish a shared beat)")
    assert max_mult > 4

    print("\n" + "=" * 70)
    print("RESULT: the declared graph wave supports conserved energy and")
    print("detector beats on this finite observation window. Pressure passages")
    print("and spectral multiplicity are separate diagnostics. A derived")
    print("autonomous TNFR moving-pattern law remains outside this experiment.")
    print("=" * 70)


if __name__ == "__main__":
    main()
