"""Re-auditing the map to particles: does a fixed-charge species carry
charge-DECOUPLED internal labels? (Layers 2 -> 3, correcting an under-count.)

THE CHALLENGE (theory creator): benchmarks/emergent_mass_charge_spectrum.py
concluded that mass is a FUNCTION OF the topological charge (m ~ W^2), so a
same-charge mass tower (e, mu, tau) cannot emerge -- and that the substrate
lacks a charge-decoupled flavour label. That conclusion measured only ONE
channel (the phase winding). It UNDER-COUNTED the substrate's degrees of
freedom. This benchmark re-audits the full chain from a TNFR start to a
localized species and shows the missing labels ARE derivable.

THE DUAL-LEVER CHANNELS (AGENTS.md): the nodal state splits into distinct
canonical channels -- phase (theta), form (EPI), capacity (nu_f) -- plus the
symplectic substrate's two sectors (geometric zeta^A = K_phi + iJ_phi,
potential zeta^B = Phi_s + iJ_dNFR) carrying a U(2) polarization. A localized
coherent structure can carry an INDEPENDENT label in each channel:

  - PHASE channel   -> the topological winding W in Z (the charge; prior work).
  - FORM channel    -> the internal standing-mode index n, with frequency
    omega_n = sqrt(lambda_n) of L_sym (the "stage" of EMERGENT_ONTOLOGY Sec.7.1a).
    This is set by CONNECTIVITY, not phase -- so it is EXACTLY decoupled from W.
  - SUBSTRATE U(2)  -> the conserved Stokes / Poincare-sphere polarization
    (symplectic_substrate.verify_polarization_symmetry) -- a spin/isospin-like
    internal 2-sphere, conserved along the flow, independent of W.

So the correct species label is (W, n, polarization, sign W) -- NOT (W, sign).
The mass has a PHASE-channel part (the winding self-energy ~ W^2) AND a
FORM-channel part (the internal excitation omega_n). At FIXED charge, the
internal index n gives a SAME-CHARGE MASS TOWER -- exactly the structure the
lepton generations need, which the earlier one-channel measurement missed.

WHAT EMERGES (measured):
  - M1: the internal frequency tower omega_n = sqrt(lambda_n) is a discrete
    ladder on a bounded structure (a real "stage" of internal states).
  - M2: overlaying a winding W = 0,1,2,3 (a PHASE change) leaves the tower
    {omega_n} IDENTICAL to ~1e-15 -- the charge (phase channel) and the internal
    tower (form/connectivity channel) are DIFFERENT canonical channels, so the
    label n is EXACTLY decoupled from the charge W.
  - M3: therefore mass = m(W, n): at FIXED W=1 the internal ladder gives distinct
    rest energies (a same-charge mass tower). The earlier "mass = f(charge)" was
    an artifact of exciting only the phase channel.
  - M4: the U(2) Stokes polarization gives THREE conserved charges -- a
    formally-available second internal label (spin/isospin-like). HONEST: the
    symmetric structures tested realize the TRIVIAL P=0 point, so a non-
    trivially-polarized species is NOT demonstrated here; the form-channel
    tower (M1-M3) is the solid charge-decoupled label.

THE CORRECTED CONCLUSION: the charge-decoupled flavour/generation label is NOT
absent -- it is DERIVABLE (the form-channel internal tower omega_n; the U(2)
polarization gives further conserved charges, formally available). The map from
a TNFR start to a species was under-counted; the fuller label is (W_phase,
n_form, sign), with the U(2) Stokes charges a formally-available extra. What
remains genuinely OPEN is narrower and sharper: the internal tower is INFINITE
(no principle selects exactly 3 generations) and its ratios (omega_n ~ n on a
box: 1, 2, 3, ...) do NOT match the real generation masses (1 : 207 : 3477). The
STRUCTURE of a generation label emerges; the specific SPECTRUM does not.

HONEST SCOPE: the standing-mode tower (omega_n = sqrt(lambda_n)) is standard
spectral graph theory / the particle-in-a-box; the U(2) Stokes polarization is
classical wave polarization (Stokes 1852, Poincare 1892), NOT a quantum spin
(no superposition or entanglement -- the doublet is per-node, the global state a
product). The TNFR content is the corrected COUNT of independent structural
labels a species carries, and that a same-charge mass tower is structurally
available. It does NOT reproduce the real lepton/quark spectrum (count 3, the
mass ratios) -- that stays OPEN (theory/EMERGENT_ONTOLOGY.md Sec.9.1). Closes no
open problem; it corrects an under-count and sharpens the open frontier.

Run:
    python benchmarks/emergent_internal_quantum_numbers.py

Theoretical anchor: AGENTS.md (dual-lever channels: phase/form/nu_f; emergent
symplectic substrate; U(2) polarization); theory/EMERGENT_ONTOLOGY.md Sec.7.1
(stage/occupant), Sec.9.1 (OPEN properties); benchmarks/emergent_particle_
catalog.py, emergent_mass_charge_spectrum.py (the one-channel predecessors);
emergent_substrate_symmetry.py (the U(2) cardinals).
Status: RESEARCH (map re-audit / under-count correction; emergence falsifier).
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tnfr.physics.emergent_particles import winding_number, winding_ring  # noqa: E402
from tnfr.physics.structural_diffusion import (  # noqa: E402
    symmetric_normalized_laplacian,
)
from tnfr.physics.symplectic_substrate import (  # noqa: E402
    verify_polarization_symmetry,
)
from tnfr.sdk import TNFR  # noqa: E402


def internal_tower(G, *, k: int = 6) -> np.ndarray:
    """Internal standing-mode frequencies omega_n = sqrt(lambda_n) of L_sym.

    The FORM-channel spectrum of the structure: a discrete ladder of internal
    excitation frequencies, set by the connectivity (not the phase).
    """
    _, lap = symmetric_normalized_laplacian(G)
    lam = np.linalg.eigvalsh(np.asarray(lap, dtype=float))
    lam = np.clip(lam, 0.0, None)
    return np.sqrt(lam)[:k]


def distinct_levels(tower, *, tol: float = 1e-6) -> list[tuple[float, int]]:
    """Collapse degenerate eigenfrequencies into distinct internal levels.

    Returns (frequency, multiplicity) pairs above the uniform vacuum. On a ring
    each level is 2-fold degenerate (the cos/sin pair) -- a real structural
    multiplicity, distinct from the level's frequency.
    """
    levels: list[list[float]] = []
    for w in tower:
        if w < tol:
            continue  # skip the uniform vacuum mode omega_0 = 0
        if not levels or abs(w - levels[-1][0]) > tol:
            levels.append([float(w), 1])
        else:
            levels[-1][1] += 1
    return [(f, int(m)) for f, m in levels]


def main() -> None:
    print("=" * 74)
    print("INTERNAL QUANTUM NUMBERS -- re-auditing the map TNFR -> particle")
    print("=" * 74)

    n = 60

    # -- M1: the internal frequency tower (the FORM channel) -------------------
    print("\n[M1] INTERNAL TOWER (form channel): omega_n = sqrt(lambda_n).")
    tower = internal_tower(winding_ring(n, 0), k=7)
    print("     first omega_n :", ", ".join(f"{w:.4f}" for w in tower))
    ladder = tower[1:]  # drop the uniform mode omega_0 = 0
    gaps = np.diff(ladder)
    print(f"     omega_0 = {tower[0]:.2e} (uniform vacuum); the rest is a ladder.")
    print(f"     distinct internal excitation modes above the vacuum: {len(ladder)}")
    assert tower[0] < 1e-6 and ladder[0] > 1e-3, "no internal tower found"
    assert np.all(gaps > -1e-9), "tower not ordered"
    print("     -> a real discrete ladder of internal states (the 'stage').")

    # -- M2: the tower is EXACTLY decoupled from the charge W ------------------
    print("\n[M2] DECOUPLING: overlay winding W=0..3 (a PHASE change) -> same tower.")
    base = internal_tower(winding_ring(n, 0), k=7)
    print(f"     {'W':>3} {'winding(meas)':>14} {'max|tower-tower_0|':>20}")
    max_dev_all = 0.0
    for w in (0, 1, 2, 3):
        G = winding_ring(n, w)
        w_meas, _ = winding_number(G)
        tw = internal_tower(G, k=7)
        dev = float(np.max(np.abs(tw - base)))
        max_dev_all = max(max_dev_all, dev)
        print(f"     {w:>3} {w_meas:>+14d} {dev:>20.2e}")
    assert max_dev_all < 1e-12, "tower changed with W -- not decoupled"
    print("     -> the charge (phase channel) leaves the internal tower")
    print("        (form/connectivity channel) EXACTLY unchanged: n _|_ W.")

    # -- M3: therefore mass = m(W, n) -- a SAME-CHARGE mass tower --------------
    print("\n[M3] SAME-CHARGE MASS TOWER: fix W=1, climb the internal index n.")
    # phase-channel (charge) self-energy ~ W^2 (prior benchmark); form-channel
    # (internal) excitation omega_n. Both independent -> mass depends on (W, n).
    w_fixed = 1
    charge_energy = (2.0 * math.pi * w_fixed / n) ** 2 * n  # ~ W^2 winding energy
    levels = distinct_levels(internal_tower(winding_ring(n, w_fixed), k=9))[:3]
    hdr = f"     {'state':>12} {'charge W':>9} {'omega_n':>9} {'mult':>5} {'~mass':>9}"
    print(hdr)
    masses = []
    for lvl, (omega_n, mult) in enumerate(levels, start=1):
        mass = charge_energy + omega_n
        masses.append(mass)
        print(
            f"     {'(W=1,n=' + str(lvl) + ')':>12} {w_fixed:>+9d} "
            f"{omega_n:>9.4f} {mult:>5} {mass:>9.4f}"
        )
    assert masses[0] < masses[1] < masses[2], "no distinct mass tower at fixed W"
    print("     -> same charge W=1, three DISTINCT masses (n=1,2,3), each a")
    print("        2-fold-degenerate level (a spin-like multiplicity).")

    # -- M4: the U(2) polarization -- a formally-available second label --------
    print("\n[M4] U(2) POLARIZATION: conserved Stokes charges (a formal 2nd label).")
    net = TNFR.create(24).ring().evolve(3)
    cert = verify_polarization_symmetry(net.G)
    print(f"     Stokes vector P = ({cert.p_1:.4f}, {cert.p_2:.4f}, {cert.p_3:.4f})")
    print(f"     su(2) algebra closes : {cert.su2_algebra_closes}")
    print(f"     charges conserved    : {cert.charges_conserved}")
    print(f"     U(2) valid           : {cert.is_valid_polarization_symmetry}")
    assert cert.is_valid_polarization_symmetry, "U(2) polarization did not verify"
    print("     -> the U(2) gives THREE conserved Stokes charges (P_1,P_2,P_3),")
    print("        a formally-available internal label. HONEST: the symmetric")
    print("        structures tested sit at the TRIVIAL P=0 point, so a non-")
    print("        trivially-polarized species is NOT demonstrated here -- the")
    print("        form-channel tower (M1-M3) is the solid decoupled label.")

    print("\n" + "=" * 74)
    print("CORRECTED MAP (TNFR start -> species):")
    print("  solid label = (W phase-winding, n internal-mode, sign W); the U(2)")
    print("    Stokes charges are a formally-available 3rd label (trivial here).")
    print("  mass  = charge part (~W^2, phase channel) + internal omega_n (form).")
    print("  => a SAME-CHARGE mass tower IS derivable (the earlier negative was an")
    print("     under-count: it excited only the phase channel).")
    print("STILL OPEN (sharper): no principle selects 3 generations, and the tower")
    print("  ratios (omega_n ~ 1,2,3,...) != the real masses (1:207:3477).")
    print("=" * 74)


if __name__ == "__main__":
    main()
