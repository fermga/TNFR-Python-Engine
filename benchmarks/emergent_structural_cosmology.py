"""Compare scoped large-scale diagnostics from separate TNFR graph models.

The script places four finite observations beside one another:

M1. Dirichlet energy decreases under fixed pure-EPI diffusion.
M2. A selected same-sign component count falls in this seeded diffusion run.
M3. The number of graph-wave amplitudes above a declared threshold grows over
    selected times on a chain.
M4. The sampled passive diffusion endpoint remains on its relaxation trajectory.

These observations do not form a cosmological model or one canonical engine
trajectory. The graph-wave matrix function has instantaneous analytic tails, so
M3 is a threshold-defined front rather than a causal horizon. The script does
not execute a driven comparator and makes no claim about physical time,
thermodynamics, spacetime, expansion, or the fate of a physical system.

Run:
    python benchmarks/emergent_structural_cosmology.py

Anchor: theory/EMERGENT_ONTOLOGY.md sections 2.5, 4.4 and 5.1.
Status: RESEARCH COMPARISON.
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
    """Return the selected fixed-graph Dirichlet functional."""
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
    print("SCOPED STRUCTURAL-HISTORY DIAGNOSTICS")
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
    degrees = np.array([G.degree[node] for node in nodes], dtype=float)
    epi -= float(np.dot(degrees, epi) / np.sum(degrees))

    # -- M1 + M2: fixed-diffusion relaxation and one domain readout -----------
    print(f"\n[M1+M2] diffusion scale 1/(nu_f*lambda_2) = {1/(nu_f*lam2):.1f};")
    print("        Dirichlet decay and a seeded same-sign component count:")
    print(f"     {'model-time':>11} {'F':>12} {'sign domains':>17}")
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
    ), "Dirichlet energy increased under the fixed diffusion step"
    assert dom_series[-1] < dom_series[0], "seeded sign-domain count did not fall"
    print("     -> PASS: F decreases under the fixed diffusion model.")
    print(
        f"        The selected sign-domain count falls from {dom_series[0]} "
        f"to {dom_series[-1]} in this seeded run; this is not a general theorem."
    )

    # -- M3: threshold-defined support in the auxiliary graph wave ------------
    print("\n[M3] THRESHOLD-DEFINED GRAPH-WAVE FRONT (chain n=60):")
    chain = nx.path_graph(60)
    _, ls = symmetric_normalized_laplacian(chain)
    w, V = np.linalg.eigh(np.asarray(ls))
    u0 = np.zeros(60)
    u0[30] = 1.0
    c = 1.0
    print(f"     {'model-time':>10} {'nodes with |u| > 0.01':>24}")
    reached = []
    for t in (2, 5, 10, 20):
        ut = V @ (np.cos(c * np.sqrt(np.clip(w, 0, None)) * t) * (V.T @ u0))
        r = int(np.sum(np.abs(ut) > 0.01))
        reached.append(r)
        print(f"     {t:>10} {r:>24d}")
    assert (
        reached == sorted(reached) and reached[-1] > reached[0]
    ), "threshold-defined support did not grow at the sampled times"
    print("     -> PASS: above-threshold support grows at the sampled times.")
    print("        The exact finite-graph wave has instantaneous analytic tails;")
    print("        this readout is not a causal cone or an expansion observable.")

    # -- M4: one passive endpoint; no driven model is executed ----------------
    print("\n[M4] PASSIVE DIFFUSION ENDPOINT")
    F_end = dirichlet_energy(G, nodes, epi)
    dom_end = coherent_domains(G, nodes, epi)
    print(f"     sampled F = {F_end:.4f}; sign domains = {dom_end}")
    print("     This remains a finite-time sample, not a physical-system fate.")
    print("     No driven comparator is executed by this benchmark.")

    print("\n" + "=" * 74)
    print("ESTABLISHED SCOPE")
    print("  fixed-diffusion Dirichlet decay; one seeded sign-domain reduction;")
    print("  one threshold-defined graph-wave front; one passive endpoint.")
    print("  These are separate graph-model diagnostics, not cosmology, causality,")
    print("  thermodynamics, or one canonical TNFR trajectory.")
    print("=" * 74)

if __name__ == "__main__":
    main()
