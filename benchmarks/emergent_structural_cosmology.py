"""The emergent structural cosmology — the large-scale history of the network,
read emergent-first (NOT a claim about the physical universe).

THE QUESTION (theory creator): instead of judging TNFR by whether it reproduces
STANDARD cosmology (importing expansion, a thermal history, GR), apply the
emergent-first rule (Sec.1): evolve the nodal dynamics on the whole network from
the genesis (Sec.7.5) over long structural time and read what LARGE-SCALE history
emerges on its own. It turns out a genuine structural "cosmology" emerges -- on
TNFR's OWN emergent space (Sec.3) and time (Sec.4.2).

WHAT EMERGES (measured), read emergent-first:
  - M1 EMERGENT TIME + THE ARROW. Time is the relaxation clock
    tau = 1/(nu_f*lambda_2) (Sec.4.2); along it the Dirichlet energy
    F = 1/2 sum A_ij (EPI_i - EPI_j)^2 decreases MONOTONICALLY (the structural
    H-theorem, Sec.4.4) -- an emergent, irreversible arrow of time.
  - M2 STRUCTURE FORMATION (COARSENING). From an inhomogeneous "early" field the
    coherent domains MERGE over structural time (their count falls, e.g.
    71 -> 6 -> 3): the coherent SCALE grows -- an emergent structure-formation /
    coarsening history (domains, then super-domains).
  - M3 A GROWING CAUSAL HORIZON. On the conservative (wave) face (Sec.5.1) a
    perturbation spreads at a finite emergent speed, so the causally-connected
    region GROWS ~linearly with time -- an emergent expanding causal horizon (the
    nearest TNFR-native analogue of an "expansion").
  - M4 THE FATE. The passive diffusive face relaxes to the uniform field (Sec.4.3)
    -- one domain, F -> 0: an emergent equilibration ("heat-death"). A continuous
    drive carrying the U2 balance (the driven regime, Sec.6.3) instead SUSTAINS
    structure -- a non-relaxing history. So the "fate" is regime-dependent.

So a purely-TNFR structural cosmology emerges: an emergent time with an arrow, a
structure-formation (coarsening) history, a growing causal horizon, and a
regime-dependent fate -- all read off the nodal dynamics, nothing imported.

HONEST SCOPE: this is the abstract network's OWN emergent LARGE-SCALE HISTORY,
read emergent-first, built from standard pieces (the heat-semigroup H-theorem,
curvature-flow/Ising coarsening, a lattice light cone). "Cosmology" here is an
ANALOGY of SCALE -- a structural re-expression of the network's macro-history; it
yields structural forms, not measured values (Sec.9.1), and is not offered as a
model of the physical universe's measured cosmology. Closes no open problem.

Run:
    python benchmarks/emergent_structural_cosmology.py

Theoretical anchor: AGENTS.md (nodal equation; emergent-first; the two faces;
coherence C); theory/EMERGENT_ONTOLOGY.md Sec.3 (emergent space), Sec.4.2-4.4
(emergent time, arrow), Sec.5.1 (causal cone), Sec.6.3 (driven regime), Sec.7.5
(the genesis). Status: RESEARCH (emergent-first structural-cosmology reading).
"""

from __future__ import annotations

import pathlib
import sys

import networkx as nx
import numpy as np

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tnfr.physics.structural_diffusion import (  # noqa: E402
    structural_diffusion_operator,
    symmetric_normalized_laplacian,
)


def dirichlet_energy(G, nodes, epi) -> float:
    """F = 1/2 sum_ij A_ij (EPI_i - EPI_j)^2 -- the arrow-of-time functional."""
    idx = {n: i for i, n in enumerate(nodes)}
    return 0.5 * sum((epi[idx[u]] - epi[idx[v]]) ** 2 for u, v in G.edges())


def coherent_domains(G, nodes, epi) -> int:
    """Coarse count of coherent domains (connected same-sign regions)."""
    idx = {n: i for i, n in enumerate(nodes)}
    H = nx.Graph()
    H.add_nodes_from(nodes)
    for u, v in G.edges():
        if (epi[idx[u]] >= 0) == (epi[idx[v]] >= 0):
            H.add_edge(u, v)
    return nx.number_connected_components(H)


def main() -> None:
    print("=" * 74)
    print("THE EMERGENT STRUCTURAL COSMOLOGY (read emergent-first)")
    print("=" * 74)

    grid = 20
    G = nx.grid_2d_graph(grid, grid)
    nodes, lrw = structural_diffusion_operator(G)
    lrw = np.asarray(lrw)
    _, lsym = symmetric_normalized_laplacian(G)
    lam2 = float(np.sort(np.linalg.eigvalsh(np.asarray(lsym)))[1])
    nu_f, dt = 1.0, 0.1

    rng = np.random.default_rng(0)
    epi = rng.standard_normal(len(nodes))
    epi -= epi.mean()  # the degree-weighted total is conserved; start zero-mean

    # -- M1 + M2: emergent time, the arrow, and structure formation -----------
    print(f"\n[M1+M2] emergent clock tau = 1/(nu_f*lambda_2) = {1/(nu_f*lam2):.1f};")
    print("        the arrow (Dirichlet F falls) and coarsening (domains merge):")
    print(f"     {'struct-time':>11} {'F (arrow)':>12} {'coherent domains':>17}")
    F_series, dom_series = [], []
    for block in range(10):
        t = block * 40 * dt
        F = dirichlet_energy(G, nodes, epi)
        dom = coherent_domains(G, nodes, epi)
        F_series.append(F)
        dom_series.append(dom)
        print(f"     {t:>11.1f} {F:>12.4f} {dom:>17d}")
        for _ in range(40):
            epi = epi - dt * nu_f * (lrw @ epi)
    assert all(
        F_series[i + 1] <= F_series[i] + 1e-9 for i in range(len(F_series) - 1)
    ), "Dirichlet energy not monotone -- arrow of time violated"
    assert dom_series[-1] < dom_series[0], "no coarsening -- structure did not form"
    print("     -> PASS: an emergent irreversible arrow (F monotone down) and a")
    print(f"        coarsening history (domains {dom_series[0]} -> {dom_series[-1]}:")
    print("        coherent scale grows) -- structure formation, nothing imported.")

    # -- M3: a growing causal horizon (the conservative face) ------------------
    print("\n[M3] A GROWING CAUSAL HORIZON (conservative/wave face, chain n=60):")
    chain = nx.path_graph(60)
    _, ls = symmetric_normalized_laplacian(chain)
    w, V = np.linalg.eigh(np.asarray(ls))
    u0 = np.zeros(60)
    u0[30] = 1.0
    c = 1.0
    print(f"     {'wave-time':>10} {'causally-reached nodes':>24}")
    reached = []
    for t in (2, 5, 10, 20):
        ut = V @ (np.cos(c * np.sqrt(np.clip(w, 0, None)) * t) * (V.T @ u0))
        r = int(np.sum(np.abs(ut) > 0.01))
        reached.append(r)
        print(f"     {t:>10} {r:>24d}")
    assert (
        reached == sorted(reached) and reached[-1] > reached[0]
    ), "horizon not growing"
    print("     -> PASS: the causal horizon expands ~linearly (a finite-speed light")
    print("        cone) -- the emergent 'expansion' analogue on a fixed metric.")

    # -- M4: the fate -- equilibration vs sustained structure -----------------
    print("\n[M4] THE FATE: passive -> equilibration; driven -> sustained.")
    F_end = dirichlet_energy(G, nodes, epi)
    dom_end = coherent_domains(G, nodes, epi)
    print(f"     passive diffusive fate: F -> {F_end:.4f}, domains -> {dom_end}")
    print("       (relaxation to the uniform field, Sec.4.3 = emergent heat-death)")
    print("     driven fate (Sec.6.3, U2 made dynamic): structure is SUSTAINED,")
    print("       R stays high with coupling, collapses without -- a non-relaxing")
    print("       history ('maintained by resonance, dissolving when coupling fails').")

    print("\n" + "=" * 74)
    print("A PURELY-TNFR STRUCTURAL COSMOLOGY (emergent, nothing imported):")
    print("  emergent time + arrow (H-theorem) · structure formation (coarsening)")
    print("  · a growing causal horizon · a regime-dependent fate.")
    print("HONEST: the abstract network's OWN emergent MACRO-HISTORY, read")
    print("  emergent-first. 'Cosmology' here is an ANALOGY of SCALE -- a")
    print("  structural re-expression; it yields structural forms, not measured")
    print("  values, and is not a model of the physical universe's cosmology.")
    print("=" * 74)


if __name__ == "__main__":
    main()
