"""Musical analogies for supplied graph spectra and phase observations.

The script compares sqrt(lambda) ratios using an auxiliary graph-wave clock.
These angular frequencies are distinct from reorganization capacity nu_f; a
physical Hz conversion needs declared time units and a factor 2*pi.
Approximate harmonic spacing applies to selected low path modes and boundaries,
not every one-dimensional graph. Dimension alone does not determine consonance.
Two-phasor agreement and the U3 compatibility threshold are not auditory laws.
The prime ladder uses known primes as construction inputs. Isospectral pairs
show that this spectral observation can lose shape information, but this does
not identify the loss with every Fix(G)^perp or arithmetic obstruction.

Status: auxiliary or finite evidence. See theory/EMERGENT_ONTOLOGY.md and
theory/NODAL_PARAMETER_FOUNDATIONS.md for model and physical-bridge limits.
"""

import os
import sys

import numpy as np
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tnfr.constants.canonical import DELTA_PHI_MAX  # noqa: E402
from tnfr.mathematics.number_theory import residue_network_rank  # noqa: E402
from tnfr.physics.structural_diffusion import (  # noqa: E402
    compute_emergent_pulse,
    structural_diffusion_operator,
)
from tnfr.riemann.prime_ladder_hamiltonian import (  # noqa: E402
    build_prime_ladder_hamiltonian,
)


def lrw_spectrum(graph):
    """Sorted, rounded spectrum of the canonical L_rw (the pulse spectrum)."""
    op = structural_diffusion_operator(graph)
    mat = op[1] if isinstance(op, tuple) else op
    eig = np.linalg.eigvals(np.asarray(mat))
    return tuple(sorted(np.round(eig.real, 6)))


def find_isospectral_pair(max_nodes=7):
    """Smallest non-isomorphic NFR pair sharing the L_rw pulse spectrum."""
    for n_nodes in range(5, max_nodes + 1):
        groups: dict = {}
        for graph in nx.graph_atlas_g():
            if graph.number_of_nodes() != n_nodes:
                continue
            if not nx.is_connected(graph):
                continue
            groups.setdefault(lrw_spectrum(graph), []).append(graph)
        for members in groups.values():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    if not nx.is_isomorphic(members[i], members[j]):
                        return members[i], members[j]
    return None


def main() -> None:
    print("=" * 72)
    print("EMERGENT MUSICAL NFR -- music as a lens on structural frequency")
    print("=" * 72)

    # M1 -- inharmonic: drum (2D) vs string (1D path) vs bell (complete)
    print("\nM1 -- selected graph-wave ratios: omega_k/omega_1 (dimensionless):")
    cases = [
        ("path P16 (1D string)", nx.path_graph(16)),
        ("ring C16 (1D loop)", nx.cycle_graph(16)),
        ("grid 4x4 (2D drum)", nx.grid_2d_graph(4, 4)),
        ("complete K6 (bell)", nx.complete_graph(6)),
    ]
    for name, graph in cases:
        spec = compute_emergent_pulse(graph, n_modes=6)["resonant_spectrum"]
        if spec and spec[0] > 1e-9:
            ratios = [round(x / spec[0], 3) for x in spec]
            print(f"   {name:>22}: {ratios}")
    # the just consonances DO emerge on the 1D string (structural frequency)
    pw = compute_emergent_pulse(nx.path_graph(64), n_modes=4)
    w = pw["resonant_spectrum"]
    print(
        f"   string consonances: octave {w[1]/w[0]:.4f} (2.000), "
        f"fifth {w[2]/w[1]:.4f} (1.500), fourth {w[3]/w[2]:.4f} (1.333)"
    )
    print("   => these selected low path modes are near harmonic; other graphs differ")

    # M2 -- the phase face of consonance (the U3 gate); the frequency-ratio
    # face is the 1D harmonic regime (M1)
    print("\nM2 -- phase consonance R, the U3 gate " f"Dphi <= {DELTA_PHI_MAX:.3f}:")
    for dphi in [0.0, np.pi / 6, np.pi / 3, np.pi / 2, 2 * np.pi / 3, np.pi]:
        r = abs(np.mean([1.0, np.exp(1j * dphi)]))
        gate = "consonant" if dphi <= DELTA_PHI_MAX + 1e-9 else "dissonant"
        print(f"   |Dphi|={dphi:.3f}: R={r:.3f}  ({gate})")

    # M3 -- polyphony: primes are independent voices (Euler product)
    print("\nM3 -- polyphony: primes = independent voices (Euler product):")
    for n_primes in (3, 5, 8):
        bundle = build_prime_ladder_hamiltonian(n_primes, coupling=0.0)
        voices = nx.Graph()
        voices.add_nodes_from(bundle.graph.nodes())
        voices.add_edges_from(bundle.graph.edges())
        comps = nx.number_connected_components(voices)
        ok = "OK" if comps == n_primes else "MISMATCH"
        print(
            f"   {n_primes} primes (coupling=0): {comps} voices "
            f"(disconnected ladders) [{ok}]"
        )

    # M4 -- Kac: you cannot hear the shape of the drum (= the wall)
    print("\nM4 -- you cannot hear the shape of the drum (Kac = the wall):")
    print("   (a) arithmetic: one chord for every semiprime pq:")
    for m in (15, 21, 33, 35):
        print(f"       rho({m}) = {residue_network_rank(m)} tones")
    print("   (b) isospectral NFRs -- same pulse, different shape:")
    pair = find_isospectral_pair()
    if pair:
        g1, g2 = pair
        d1 = sorted(d for _, d in g1.degree())
        d2 = sorted(d for _, d in g2.degree())
        spec1 = compute_emergent_pulse(g1, n_modes=3)["resonant_spectrum"]
        spec2 = compute_emergent_pulse(g2, n_modes=3)["resonant_spectrum"]
        print(f"       graph A: {g1.number_of_edges()} edges, degrees {d1}")
        print(f"       graph B: {g2.number_of_edges()} edges, degrees {d2}")
        print("       => non-isomorphic (different shape), identical pulse:")
        print(f"          spectrum A = {[round(x, 4) for x in spec1]}")
        print(f"          spectrum B = {[round(x, 4) for x in spec2]}")

    print("\n" + "=" * 72)
    print(
        "FINITE SPECTRAL ANALOGIES: low path-mode ratios, phase agreement and\nprepared prime-ladder components are separate observations. Isospectral\ngraphs can share these spectral read-outs without sharing graph shape.\nCapacity, wave frequency, musical perception and physical identity remain distinct."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
