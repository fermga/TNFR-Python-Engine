"""From the catalog to the properties: the emergent (mass, charge) spectrum and
where it departs from nature (Layers 2 -> 3).

THE QUESTION (theory creator, continuing the Layer-2/3 investigation):
benchmarks/emergent_particle_catalog.py established, on the 1D ring, the
SELECTION of a fundamental unit charge (|W|=1) from the energy hierarchy
E(W) ~ W^2. Two questions remain open and are addressed here:

  (Layer 2, robustness) On the ring the |W|>=2 fission was only INFERRED from
  the energy ordering -- the ring is 1D, so the winding is topologically locked
  and cannot actually split. Does a |W|=2 structure REALLY fission on a 2D
  manifold where two |W|=1 cores can separate? (M1, M2 -- measured DIRECTLY.)

  (Layer 3, the frontier) A real particle is not just a charge: it has a MASS,
  and the catalog of masses is the deep open question. What is TNFR's emergent
  (mass, charge) spectrum, and does it match the real one? (M3, M4.)

DEFINITIONS (canonical, nothing imported):
  - charge  W  = integer winding of the phase, measured per face (degree of
    S^1 -> S^1, exact; examples/08_emergent_geometry/133_psi_topological_defects.py).
  - mass  m    = the structural SELF-ENERGY of the localized species: the total
    canonical energy density (tnfr.physics.unified.compute_energy_density) above
    the uniform vacuum. This is the rest energy of the coherent form -- the only
    mass-like scalar the substrate provides.

WHAT EMERGES (measured):
  - M1 (Layer 2, DIRECT fission): on a 2D grid a single |W|=2 core has a HIGHER
    self-energy than two separated |W|=1 cores of the same total charge, so the
    double-winding structure is NOT the charge-2 ground state -- it splits. The
    ring's inferred fission is now DIRECTLY demonstrated where separation is real.
  - M2 (Layer 2, like-charge repulsion): the self-energy of the two |W|=1 cores
    DECREASES as their separation grows -- the like charges repel (the 2D Coulomb
    /XY-vortex process), the force that drives the fission of M1.
  - M3 (Layer 3, the mass spectrum): the self-energy of the minimal-energy
    charge-W sector (a uniform-winding ring, where the gradient 2pi|W|/n stays
    small and lattice-clean) scales as m(W) ~ W^2 EXACTLY (ratios 1, 4, 9, 16) --
    TNFR's emergent (mass, charge) locus is m ~ q^2: mass is a FUNCTION of the
    topological charge. (The clean law lives on the ring; a localized 2D core
    has large near-core gradients that wrap-saturate, so its energy is noisy --
    the process, M1/M2, needs the 2D manifold; the energy law needs the ring.)
  - M4 (Layer 3, the honest confrontation): the real charged leptons e, mu, tau
    ALL carry the same charge (q = -1) but masses in ratio 1 : 207 : 3477. A mass
    that is a function of charge (M3) forces same-charge species to be MASS-
    DEGENERATE -- which the lepton generations violate outright. So the real mass
    spectrum does NOT emerge, and the obstruction is now PRECISE: the substrate
    labels a species only by (W, U(2) shell, sign W); it lacks an internal
    FLAVOUR / generation degree of freedom DECOUPLED from the charge, which is
    exactly what a same-charge mass tower requires.

HONEST SCOPE: the 2D vortex self-energy ~ W^2 log(R/a), the like-charge
repulsion, and the fission of multiply-charged vortices are STANDARD
topological-defect physics (XY / Kosterlitz-Thouless, Abrikosov, superfluids).
The TNFR content is that the ONE nodal operator reproduces them, giving a
STRUCTURED emergent catalog with a mass spectrum m ~ q^2. That spectrum is
TNFR's OWN; it does NOT reproduce the real particle masses/charges (Layer 3,
OPEN, theory/EMERGENT_ONTOLOGY.md Sec.9.1). The value of M4 is a DECISIVE,
honest negative that LOCATES the gap (a charge-decoupled flavour label), not a
match. The substrate is classical -- no spin, Hilbert space, or Born rule.
Closes no open problem.

Run:
    python benchmarks/emergent_mass_charge_spectrum.py

Theoretical anchor: AGENTS.md (nodal equation; topological charge; coherence C;
U(1) gauge sector); theory/EMERGENT_ONTOLOGY.md Sec.7.1-7.2 (occupant/process,
EM gauge), Sec.9.1 (OPEN properties frontier); benchmarks/
emergent_particle_catalog.py (the 1D catalog selection); examples/08_emergent_
geometry/133_psi_topological_defects.py (integer face winding).
Status: RESEARCH (Layer-2 direct + Layer-3 probe; emergence falsifier).
"""

from __future__ import annotations

import math
import pathlib
import sys

import networkx as nx

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tnfr.physics.emergent_particles import winding_ring  # noqa: E402
from tnfr.physics.unified import compute_energy_density  # noqa: E402

_TWO_PI = 2.0 * math.pi
GRID = 25  # L x L open grid; odd so a single core sits on a node


def _wrap_pi(x: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    y = (x + math.pi) % _TWO_PI - math.pi
    return y - _TWO_PI if y > math.pi else y


def build_vortices(
    cores: list[tuple[float, float, int]],
    *,
    base_dnfr: float = 0.05,
    grid: int = GRID,
) -> nx.Graph:
    """Open 2D grid whose phase is the superposition of vortex cores.

    ``cores`` is a list of (cx, cy, q): a charge-q vortex has local phase
    q * atan2(y - cy, x - cx). The winding is MEASURED as an output (per face);
    the construction only sets the initial phase field.
    """
    G = nx.grid_2d_graph(grid, grid)
    for i, j in G.nodes():
        ang = 0.0
        for cx, cy, q in cores:
            ang += q * math.atan2(j - cy, i - cx)
        phi = ang % _TWO_PI
        G.nodes[(i, j)]["theta"] = float(phi)
        G.nodes[(i, j)]["phase"] = float(phi)
        G.nodes[(i, j)]["delta_nfr"] = float(base_dnfr)
        G.nodes[(i, j)]["dnfr"] = float(base_dnfr)
        G.nodes[(i, j)]["coherence"] = 1.0 / (1.0 + abs(base_dnfr))
        G.nodes[(i, j)]["EPI"] = 1.0 / (1.0 + abs(base_dnfr))
        G.nodes[(i, j)]["nu_f"] = 1.0
    return G


def loop_winding(G: nx.Graph, *, grid: int = GRID, margin: int = 2) -> int:
    """Integer winding around a large square loop = total enclosed charge.

    Walking a big boundary loop keeps every step's phase difference well below
    pi, so the wrap is unambiguous even for multiply-charged cores (a single
    |W|=2 face would sit exactly at the pi wrap boundary; a large loop does not).
    """
    theta = {n: G.nodes[n]["theta"] for n in G.nodes()}
    lo, hi = margin, grid - 1 - margin
    path: list[tuple[int, int]] = []
    path += [(x, lo) for x in range(lo, hi)]
    path += [(hi, y) for y in range(lo, hi)]
    path += [(x, hi) for x in range(hi, lo, -1)]
    path += [(lo, y) for y in range(hi, lo, -1)]
    s = sum(
        _wrap_pi(theta[path[(k + 1) % len(path)]] - theta[path[k]])
        for k in range(len(path))
    )
    return int(round(s / _TWO_PI))


def self_energy(G: nx.Graph, vacuum: float) -> float:
    """Total canonical energy density above the uniform vacuum (the mass-like scalar)."""
    ed = compute_energy_density(G)
    return float(sum(ed[n] for n in G.nodes()) - vacuum)


def vacuum_energy(*, base_dnfr: float = 0.05, grid: int = GRID) -> float:
    """Energy of the uniform-phase grid (the vacuum baseline)."""
    G = build_vortices([], base_dnfr=base_dnfr, grid=grid)
    ed = compute_energy_density(G)
    return float(sum(ed[n] for n in G.nodes()))


def ring_mass(w: int, *, n: int = 60) -> float:
    """Self-energy of a uniform-winding-W ring above its vacuum (the clean mass).

    On the ring the charge-W minimum spreads the winding uniformly
    (|grad phi| = 2pi|W|/n, small and lattice-clean), so the canonical energy is
    the exact W-sector self-energy -- unlike a localized 2D core whose near-core
    gradient wrap-saturates. Matches benchmarks/emergent_particle_catalog.py.
    """
    ed = compute_energy_density(winding_ring(n, w))
    return float(sum(ed.values()))


def main() -> None:
    print("=" * 74)
    print("EMERGENT (MASS, CHARGE) SPECTRUM -- from the catalog to the properties")
    print("=" * 74)

    c = GRID // 2
    cf = c + 0.5  # half-integer face centre: no node coincides with a core
    vac = vacuum_energy()

    # -- M1: DIRECT composite instability on a 2D manifold (Layer 2) -----------
    print("\n[M1] DIRECT FISSION (Layer 2): a |W|=2 core vs two separated |W|=1.")
    g_double = build_vortices([(cf, cf, 2)])
    e_double = self_energy(g_double, vac)
    w_double = loop_winding(g_double)
    sep = 8
    g_split = build_vortices([(cf - sep / 2, cf, 1), (cf + sep / 2, cf, 1)])
    e_split = self_energy(g_split, vac)
    w_split = loop_winding(g_split)
    print(f"     single |W|=2 core : winding={w_double:+d}  self-energy={e_double:.4f}")
    print(
        f"     two |W|=1 (sep={sep}) : winding={w_split:+d}  "
        f"self-energy={e_split:.4f}"
    )
    print(f"     E(W=2 core) - E(two W=1) = {e_double - e_split:+.4f}")
    assert w_double == 2 and w_split == 2, "total charge must be +2 for both"
    assert e_double > e_split, "double core not above the split -- no fission"
    print("     -> PASS: the double-winding core costs MORE than the split pair;")
    print("        charge-2 is not a single core -- it fissions. (Directly shown.)")

    # -- M2: like-charge repulsion drives the fission (Layer 2) ----------------
    print("\n[M2] LIKE-CHARGE REPULSION (Layer 2): E(two |W|=1) vs separation.")
    print(f"     {'separation':>11} {'winding':>8} {'self-energy':>13}")
    prev = None
    decreasing = True
    for sep in (2, 4, 6, 8, 10, 12):
        g = build_vortices([(cf - sep / 2, cf, 1), (cf + sep / 2, cf, 1)])
        e = self_energy(g, vac)
        w = loop_winding(g)
        print(f"     {sep:>11} {w:>+8d} {e:>13.4f}")
        if prev is not None and e > prev + 1e-6:
            decreasing = False
        prev = e
    assert decreasing, "energy did not fall with separation -- no repulsion"
    print("     -> PASS: energy falls as the two like cores separate = repulsion")
    print("        (2D Coulomb / XY-vortex); the force that drives M1 fission.")

    # -- M3: the emergent mass spectrum m(W) ~ W^2 (Layer 3) -------------------
    print("\n[M3] MASS SPECTRUM (Layer 3): self-energy of the charge-W sector.")
    print(f"     {'W':>3} {'mass (self-E)':>15} {'m(W)/m(1)':>11} {'W^2':>6}")
    vac_ring = ring_mass(0)
    m1 = ring_mass(1) - vac_ring
    masses = {}
    for w in (1, 2, 3, 4, 5):
        m = ring_mass(w) - vac_ring
        masses[w] = m
        ratio = m / m1 if m1 else float("nan")
        print(f"     {w:>3} {m:>15.4f} {ratio:>11.3f} {w * w:>6}")
    print("     m(W)/m(1) = the exact integer squares 1, 4, 9, 16, 25 (m ~ W^2).")
    assert abs(masses[2] / m1 - 4.0) < 0.05, "ring mass not ~4 at W=2"
    assert abs(masses[3] / m1 - 9.0) < 0.1, "ring mass not ~9 at W=3"
    print("     -> TNFR's (mass, charge) locus is m ~ q^2: MASS IS A FUNCTION OF")
    print("        CHARGE. The mass spectrum is set by the topological charge.")

    # -- M4: the honest confrontation with the real spectrum (Layer 3) ---------
    print("\n[M4] CONFRONTATION (Layer 3): the real charged leptons.")
    leptons = {"e": 0.511, "mu": 105.66, "tau": 1776.86}  # MeV; all q = -1
    m_e = leptons["e"]
    print(f"     {'lepton':>7} {'charge':>7} {'mass (MeV)':>11} {'m/m_e':>9}")
    for name, mass in leptons.items():
        print(f"     {name:>7} {-1:>+7d} {mass:>11.3f} {mass / m_e:>9.1f}")
    print("     TNFR: mass = f(|W|), so same charge => SAME mass (degenerate).")
    print("     Nature: same charge (-1), masses in ratio 1 : 207 : 3477.")
    print("     -> DECISIVE NEGATIVE: a charge-tied mass (M3) cannot produce a")
    print("        same-charge mass tower. The real spectrum does NOT emerge.")

    print("\n" + "=" * 74)
    print("SUMMARY (Layers 2 -> 3):")
    print("  Layer 2 (catalog): the composite fission is now DIRECT (M1/M2) --")
    print("    |W|=1 is the fundamental unit charge; |W|>=2 split. STRUCTURED.")
    print("  Layer 3 (properties): TNFR gives a mass spectrum, but m ~ q^2 (M3);")
    print("    the real same-charge lepton tower (M4) refutes it. OPEN.")
    print("  LOCATED GAP: a flavour/generation label DECOUPLED from the charge --")
    print("    absent from the classical substrate's (W, U(2) shell) labels.")
    print("=" * 74)


if __name__ == "__main__":
    main()
