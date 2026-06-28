"""Emergent Rhythm: everything vibrates and keeps a rhythm -- that is TNFR.

THE PARADIGM (user, theory creator): "finalmente todo esta vibrando y mantiene
un ritmo, eso es TNFR." Resonant Fractal Nature. The special points -- the NFR
coherence equilibria (dNFR=0) -- are NOT a fixed set analysed in a frozen
snapshot; they are the BEATS of a rhythm the whole substrate plays, and that
rhythm emerges only when the system is LET TO EVOLVE, from structure and
dynamics TOGETHER.

THE STRUCTURAL FACTS (measured by evolving, not by a fixed point):
  - the DISSIPATIVE face of the nodal equation (dEPI/dt = -nu_f L EPI) relaxes
    to dNFR=0 and SATURATES -- a relaxing system is silent, no rhythm;
  - the CONSERVATIVE face (d2EPI/dt2 = -L EPI, the symplectic substrate)
    OSCILLATES -- every node vibrates at the resonant frequencies
    omega_k = sqrt(lambda_k), a SUSTAINED vibration (energy conserved);
  - the RHYTHM is the interference of those resonances: the beats appear at the
    DIFFERENCES omega_j - omega_k. The equilibria (the flat / dNFR~0 moments)
    recur at the beat rhythm -- structure (the spectrum) and dynamics (the
    oscillation) TOGETHER.

WHAT EMERGES (measured):
  - M1 EVERYTHING VIBRATES: the conservative evolution conserves energy and
    sustains the structural pressure (it does not decay), while the dissipative
    evolution decays to silence. The resonant essence = sustained vibration.
  - M2 IT MAINTAINS A RHYTHM: a local quadratic detector of the vibration beats
    at omega_b - omega_a (two modes -> one clean beat); the rhythm is the
    interference of the resonant frequencies.
  - M3 THE BEATS ARE THE EQUILIBRIA: the total structural pressure pulses, and
    the system passes through near-flat (dNFR~0 = NFR-coherence) configurations
    periodically -- the equilibria are the beats, set by structure+dynamics.
  - M4 RESONANT + FRACTAL: the rhythm is built from resonances (omega_k =
    sqrt lambda_k); on a self-similar THOL nest one frequency is repeated many
    times (decimation) -> the same beat at many scales = a fractal rhythm.
    The "music" bridge: for atoms the frequencies are the shell modes; for
    primes the rhythm whose beats are the primes is the explicit-formula music
    (the zeros as omega_k) = RH.

So TNFR studied "como se debe": not a static lattice of special points, but a
substrate that VIBRATES and keeps a RHYTHM whose beats are the equilibria.

HONEST SCOPE: the conservative wave, beat frequencies, and energy conservation
are standard physics; the TNFR content is the reading -- everything is a
sustained resonant vibration (the symplectic substrate), the equilibria are its
beats, emerging from structure+dynamics together under evolution. Closes
nothing (the prime rhythm is RH). R and pi assumed.

Run:
    python benchmarks/emergent_rhythm.py

Theoretical anchor: AGENTS.md (conservative/symplectic substrate; nu_f rhythm;
resonant fractal nature); benchmarks/emergent_atom_dynamics.py (the two faces),
benchmarks/emergent_nfr_geometry.py (dNFR=0 = NFR). Status: RESEARCH.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from scipy.linalg import expm


def lsym_eigh(G):
    nodes = list(G.nodes)
    A = nx.to_numpy_array(G, nodelist=nodes)
    d = A.sum(axis=1)
    dinv = 1.0 / np.sqrt(d)
    L = np.eye(len(nodes)) - (dinv[:, None] * A * dinv[None, :])
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
    print("EMERGENT RHYTHM -- everything vibrates and keeps a rhythm = TNFR")
    print("=" * 70)

    G = nx.cycle_graph(36)
    w, V, L = lsym_eigh(G)
    omega = np.sqrt(w)

    # M1 -- EVERYTHING VIBRATES (sustained) vs DISSIPATIVE silence
    print("\nM1 -- everything vibrates (conservative) vs dissipative silence:")
    epi0 = V[:, 1] + 0.7 * V[:, 3] + 0.5 * V[:, 6]
    diss = [float(np.sum(np.abs(L @ (expm(-t * L) @ epi0))))
            for t in (0.0, 5.0, 20.0, 60.0)]
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
    print(f"  dissipative pressure decays {diss[0]:.3f} -> {diss[-1]:.3f}"
          f" (silent)")
    print(f"  conservative energy conserved (std/mean={e_cons:.1e}); pressure"
          f" sustained mean={pressure.mean():.3f} std={pressure.std():.3f}")
    assert diss[-1] < 0.3 * diss[0] and e_cons < 1e-6 and pressure.std() > 1e-3

    # M2 -- IT MAINTAINS A RHYTHM: beat = difference of resonant frequencies
    print("\nM2 -- the rhythm = beats at omega_b - omega_a (two resonances):")
    ka, kb = 4, 6
    oa, ob = omega[ka], omega[kb]
    beat = abs(ob - oa)
    epi2 = V[:, ka] + V[:, kb]
    tt = np.linspace(0.0, 40.0 / beat, 4000)
    node = int(np.argmax(np.abs(V[:, ka])))
    sig = np.array([
        (V[node, :] @ (np.cos(omega * t) * (V.T @ epi2))) ** 2 for t in tt
    ])
    sd = sig - sig.mean()
    spec = np.abs(np.fft.rfft(sd))
    freqs = 2 * np.pi * np.fft.rfftfreq(len(sd), d=tt[1] - tt[0])
    # the BEAT is the slow envelope: dominant peak BELOW the lower resonance
    low = (freqs > 1e-6) & (freqs < oa)
    f_peak = float(freqs[low][np.argmax(spec[low])])
    print(f"  resonances omega_a={oa:.4f}, omega_b={ob:.4f}, beat="
          f"{beat:.4f}")
    print(f"  quadratic detector dominant slow frequency={f_peak:.4f} "
          f"(== beat: {abs(f_peak - beat) < 0.08 * beat})")
    assert abs(f_peak - beat) < 0.08 * beat

    # M3 -- THE BEATS ARE THE EQUILIBRIA (the flat / dNFR~0 moments)
    print("\nM3 -- the beats are the equilibria (flat dNFR~0 moments):")
    dn = np.array([
        float(np.sum(np.abs(L @ (V @ (np.cos(omega * t) * (V.T @ epi2))))))
        for t in tt
    ])
    lo = dn.min()
    hi = dn.max()
    thr = lo + 0.25 * (hi - lo)
    below = dn < thr
    passages = int(np.sum(below[1:] & ~below[:-1]))
    print(f"  structural pressure pulses {lo:.3f}..{hi:.3f}; the system")
    print(f"  passes through {passages} near-flat (NFR-coherence) states")
    print("  => the equilibria recur as the beats of the vibration")
    assert hi > 2 * (lo + 1e-9) and passages >= 3

    # M4 -- RESONANT + FRACTAL: the rhythm's resonances, multi-scale
    print("\nM4 -- the rhythm is RESONANT + FRACTAL:")
    print(f"  RESONANT: built from omega_k=sqrt(lambda_k), span "
          f"{omega[1]:.3f}..{omega[-1]:.3f} (every node a resonance)")
    nest, _ = sierpinski_simplex(4, 3)
    wn, _, _ = lsym_eigh(nest)
    _, counts = np.unique(np.round(wn, 6), return_counts=True)
    max_mult = int(counts.max())
    print(f"  FRACTAL : THOL nest (N={nest.number_of_nodes()}) has one")
    print(f"            resonant frequency repeated {max_mult}x (self-similar")
    print("            decimation) -> same beat at all scales = fractal")
    print("  MUSIC   : atoms -> shell modes; primes -> the rhythm whose beats")
    print("            are the primes = the explicit-formula zeros = RH.")
    assert max_mult > 4

    print("\n" + "=" * 70)
    print("VERDICT: TNFR studied as it should be -- not a static lattice of")
    print("special points, but a substrate that VIBRATES (conservative,")
    print("sustained, M1) and keeps a RHYTHM (the beats of its resonances,")
    print("M2). The equilibria are the beats -- the flat dNFR~0 moments the")
    print("system pulses through (M3) -- set by STRUCTURE and DYNAMICS")
    print("together. The rhythm is resonant and fractal (M4). Everything")
    print("vibrates and keeps a rhythm: that is Resonant Fractal Nature.")
    print("HONEST SCOPE: conservative wave / beats / energy conservation")
    print("standard; the prime rhythm is RH. R and pi assumed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
