"""Separate winding, fixed-graph mode and auxiliary polarization read-outs.

The same fixed support has the same Laplacian spectrum when only prepared
phase winding changes. This establishes independence of those observations on
that construction, not independence under the full joint nodal dynamics.
The finite graph has a finite mode ladder. Assigning omega=sqrt(lambda) uses
an auxiliary graph-wave interpretation; it is not the capacity nu_f.
The displayed mass combines a winding-energy read-out and a mode frequency by
an explicit addition, with no derived units, mass law or particle identification.
A cosine/sine degeneracy is not spin. Conserved Stokes quantities belong to the
separate auxiliary isotropic Hamiltonian, not arbitrary engine operators.

Status: auxiliary or finite evidence. See theory/EMERGENT_ONTOLOGY.md and
theory/NODAL_PARAMETER_FOUNDATIONS.md for model and physical-bridge limits.
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
    print("FIXED-SUPPORT MODES, PREPARED WINDING AND AUXILIARY POLARIZATION")
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

    # -- M2: fixed support leaves the graph spectrum independent of prepared phase ------------------
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
    print(
        "        (fixed graph) unchanged in these preparations; no joint dynamical closure."
    )

    # -- M3: explicitly assigned energy-plus-mode proxy --------------
    print("\n[M3] ASSIGNED ENERGY-PLUS-MODE LADDER: fix W=1, vary n.")
    # phase-channel (charge) self-energy ~ W^2 (prior benchmark); form-channel
    # (internal) excitation omega_n. The addition is a proxy definition, not a mass law.
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
    print("     -> same prepared W=1, three DISTINCT assigned values (n=1,2,3), each a")
    print("        2-fold cosine/sine mode level; no spin identification.")

    # -- M4: the U(2) polarization -- a formally-available second label --------
    print("\n[M4] AUXILIARY U(2): the separate isotropic-model Stokes check.")
    net = TNFR.create(24).ring().evolve(3)
    cert = verify_polarization_symmetry(net.G)
    print(f"     Stokes vector P = ({cert.p_1:.4f}, {cert.p_2:.4f}, {cert.p_3:.4f})")
    print(f"     su(2) algebra closes : {cert.su2_algebra_closes}")
    print(f"     charges conserved    : {cert.charges_conserved}")
    print(f"     U(2) valid           : {cert.is_valid_polarization_symmetry}")
    assert cert.is_valid_polarization_symmetry, "U(2) polarization did not verify"
    print(
        "     -> the auxiliary isotropic model has THREE Stokes charges (P_1,P_2,P_3),"
    )
    print("        a formally-available internal label. HONEST: the symmetric")
    print("        structures tested sit at the TRIVIAL P=0 point, so a non-")
    print("        trivially-polarized species is NOT demonstrated here -- the")
    print("        fixed-support tower (M1-M3) is a separate static observation.")

    print("\n" + "=" * 74)
    print("READ-OUT SCOPE:")
    print("  Winding and fixed-support mode indices describe separate prepared data.")
    print("  Auxiliary Stokes quantities add a separately assumed model observation.")
    print("  The displayed mass formula is an assigned energy-plus-frequency sum.")
    print(
        "  It does not derive rest mass, flavour, spin or a same-charge particle tower."
    )
    print("  The finite graph supplies finitely many modes.")
    print(
        "  Dynamic closure and physical interpretation require independent derivations."
    )
    print(
        "  Three physical generations and measured mass ratios are not selected here."
    )
    print("=" * 74)


if __name__ == "__main__":
    main()
