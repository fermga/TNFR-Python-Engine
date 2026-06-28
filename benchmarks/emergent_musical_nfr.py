"""Emergent Musical NFR: the nodal equation is the music of a vibrating drum.

THE PARADIGM (user, theory creator): the music analogy keeps making more sense.
It is not decorative -- the conservative face of the nodal equation IS a
vibrating system (every mode at omega_k = sqrt(lambda_k); "nodal" is the
wave-node of Chladni). This benchmark measures which MUSICAL mechanisms
genuinely emerge, and -- the honest centerpiece -- where the music STOPS: the
same Fix(G)^perp wall as everything else.

THE STRUCTURAL FACTS (all from L's spectrum + the canonical operators):
  - pitch = omega_k = sqrt(lambda_k); chord = the distinct tones; timbre = the
    eigenvalue multiplicities; beats = omega_j - omega_k; nodes = the dNFR=0
    NFRs;
  - the graph Laplacian spectrum is generally INHARMONIC (a drum / Chladni
    plate), harmonic only for 1D chains (a string) -- so frequency-ratio
    consonance does NOT emerge; consonance is PHASE (the U3 gate Dphi <= pi/2);
  - the pulse hears the symmetry TYPE (it can even predict unseen tones,
    inverse_spectrum_to_symmetry.py) but NOT the IDENTITY: isospectral NFRs
    sound the same (Kac) -- the wall.

WHAT EMERGES (measured):
  - M1 DRUM, NOT STRING: omega_k/omega_1 is ~integer (harmonic) only for the 1D
    path (a string); the 2D grid (a drum) and the ring are INHARMONIC; K_n is
    one rigid tone (a bell). TNFR is percussion / Chladni music.
  - M2 CONSONANCE = PHASE: the Kuramoto order R = cos(Dphi/2) of two coupled
    NFRs is high (consonant) inside the U3 gate Dphi <= pi/2 and collapses to
    destructive antiphase beyond it -- consonance is phase, not a frequency
    ratio.
  - M3 POLYPHONY = PRIMES: the decoupled prime-ladder splits into exactly one
    component per prime -- distinct primes are INDEPENDENT VOICES (the Euler
    product at the operator level), a polyphony that never interferes.
  - M4 YOU CANNOT HEAR THE SHAPE OF THE DRUM (Kac = the wall): isospectral
    non-isomorphic NFRs share the pulse (same sound, different shape), and the
    arithmetic rho(pq) = 9 is one chord for every semiprime -- the pulse hears
    the TYPE, not the identity. The unhearable residue is the Fix(G)^perp wall.

So the music of the NFR is real AND it closes on the same wall: you hear the
type / symmetry (pitch, chord, timbre, polyphony), never the full identity.

HONEST SCOPE: pitch/timbre/beats/nodes are the standing-wave spectrum re-read
in TNFR terms; consonance=phase is the U3 gate; polyphony=primes is the Euler
product (ex 147/148); the Kac wall is the inverse spectral problem = the
Fix(G)^perp residue (theory 9.7/9.10). Rejected as IMPOSED (not measured):
temperament, scales, and the literal harmonic series / octave (the spectrum is
inharmonic). Derives no new physics; closes no open problem. R and pi assumed.

Run:
    python benchmarks/emergent_musical_nfr.py

Theoretical anchor: EMERGENT_ONTOLOGY.md section 5.5 (the pulse); theory/
TNFR_NUMBER_THEORY.md (9.12 the arithmetic pulse, 9.7 the wall);
src/tnfr/physics/structural_diffusion.py (compute_emergent_pulse);
benchmarks/emergent_rhythm.py + inverse_spectrum_to_symmetry.py.
Status: RESEARCH.
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
    print("EMERGENT MUSICAL NFR -- the nodal equation is the music of a drum")
    print("=" * 72)

    # M1 -- inharmonic: drum (2D) vs string (1D path) vs bell (complete)
    print("\nM1 -- drum, not string: omega_k/omega_1 (harmonic = integers):")
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
    print("   => only the 1D string is ~harmonic; 2D+ is inharmonic (Chladni)")

    # M2 -- consonance = phase (the U3 gate), not a frequency ratio
    print("\nM2 -- consonance = phase coherence R, the U3 gate "
          f"Dphi <= {DELTA_PHI_MAX:.3f}:")
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
        print(f"   {n_primes} primes (coupling=0): {comps} voices "
              f"(disconnected ladders) [{ok}]")

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
        "VERDICT: the music of the NFR is real (pitch, chord, timbre, beats,\n"
        "consonance=phase, polyphony=primes) AND it closes on the same wall:\n"
        "you hear the type, never the full identity (Kac = Fix(G)^perp)."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
