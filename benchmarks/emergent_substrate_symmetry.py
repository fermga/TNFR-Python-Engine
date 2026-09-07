"""Auxiliary-substrate U(2) and oscillator-cardinal correspondence.

The symplectic substrate model declares two conjugate sectors per node,

    zeta^A = K_phi + i J_phi   (geometric sector)
    zeta^B = Phi_s + i J_dnfr  (potential sector)

Its exact isotropic harmonic flow has U(2) polarization symmetry. This module
checks that implemented auxiliary flow on fields extracted from a ring and then
compares standard U(d) oscillator dimensions, with a selected factor-two
capacity, against familiar shell-number sequences:

  d = 2 (the substrate)  -> 2, 6, 12, 20  (2D quantum dots)
  d = 3                  -> 2, 8, 20, 40  (3D nuclear oscillator)

The arithmetic equality is a correspondence after choosing the auxiliary
two-sector Hamiltonian and capacity convention. It does not derive quantum-dot
or nuclear shell physics, prove that the nodal equation generates the substrate,
certify any of the 13 engine operators as symplectic, or exclude alternative
auxiliary models. A dynamical bridge and an independently inferred effective
dimension remain open.

Run:
    python benchmarks/emergent_substrate_symmetry.py

Theoretical anchor: AGENTS.md (emergent symplectic substrate; U(2));
symplectic_substrate.py (H_sub = sum of decoupled oscillators);
benchmarks/emergent_shell_cardinals.py (the symmetry-cardinal chain).
Status: RESEARCH (scoped correspondence check).
"""

from __future__ import annotations

import pathlib
import sys
from math import comb

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tnfr.physics.symplectic_substrate import (  # noqa: E402
    BLOCK_SYMPLECTIC_FORM,
    CONJUGATE_PAIR_LABELS,
    extract_phase_space_point,
    verify_polarization_symmetry,
)
from tnfr.sdk import TNFR  # noqa: E402

QUANTUM_DOT_2D = [2, 6, 12, 20, 30]  # comparison sequence
NUCLEAR_OSC_3D = [2, 8, 20, 40, 70]  # comparison sequence


def u_d_cardinals(d: int, n: int = 5) -> list[int]:
    """Cumulative shell cardinals of a U(d) isotropic oscillator.

    Level N has the symmetric U(d) irrep, dimension C(N+d-1, d-1); the
    factor 2 is a declared capacity convention for this comparison.
    """
    out: list[int] = []
    total = 0
    for N in range(n):
        total += 2 * comb(N + d - 1, d - 1)
        out.append(total)
    return out


def main() -> None:
    print("=" * 70)
    print("AUXILIARY SUBSTRATE U(2) AND CARDINAL CORRESPONDENCE")
    print("=" * 70)

    # -- M1: verify the declared auxiliary U(2) flow on extracted fields -----
    net = TNFR.create(24).ring().evolve(3)
    cert = verify_polarization_symmetry(net.G)
    print("\n[M1] Auxiliary U(2) flow initialized from ring fields:")
    print(f"     su(2) algebra closes : {cert.su2_algebra_closes}")
    print(f"     charges conserved    : {cert.charges_conserved}")
    print(f"     full polarization    : {cert.full_polarization_holds}")
    print(f"     U(2) valid           : {cert.is_valid_polarization_symmetry}")
    assert cert.is_valid_polarization_symmetry, "substrate U(2) did not verify"
    print("     -> PASS: the declared two-sector harmonic model carries U(2).")
    print("        This verifies the auxiliary flow, not its derivation from")
    print("        the engine's nodal evolution.")

    # -- M2: U(2) cardinals = 2D quantum-dot magic numbers ------------------
    u2 = u_d_cardinals(2)
    print("\n[M2] U(2) isotropic-oscillator cardinals (cumulative 2(N+1)):")
    print(f"     U(2) level dims (N+1)  : {[N + 1 for N in range(5)]}")
    print(f"     U(2) magic numbers     : {u2}")
    print(f"     comparison sequence    : {QUANTUM_DOT_2D}")
    assert u2 == QUANTUM_DOT_2D, u2
    print("     -> PASS: the selected U(2) comparison sequence is 2, 6, 12, 20.")
    print("        Equality with the listed quantum-dot counts is a correspondence.")

    # -- M3: the sector count d sets the family; the frontier is the 3rd ----
    u3 = u_d_cardinals(3)
    print("\n[M3] The cardinal family is set by the sector count d = U(d):")
    print(f"     d=2 (substrate)        : {u2}   2D quantum dots")
    print(f"     d=3 (third sector)     : {u3}   3D nuclear oscillator")
    assert u3 == NUCLEAR_OSC_3D, u3
    print("     -> PASS: U(3) gives the nuclear 2, 8, 20; the substrate has")
    print("        two sectors by construction. A physical dimension does not")
    print("        follow from this cardinal comparison alone.")

    # -- M4: can a THIRD conjugate sector emerge? (structural lock: NO) ------
    pt = extract_phase_space_point(net.G)
    print("\n[M4] Can a THIRD conjugate sector emerge (U(2) -> U(3))?")
    print(f"     conjugate sectors/node : {len(CONJUGATE_PAIR_LABELS)} "
          f"{CONJUGATE_PAIR_LABELS}")
    print(f"     symplectic block       : {BLOCK_SYMPLECTIC_FORM.shape} "
          "= 4 dims/node = 2 pairs")
    print("     |grad phi| (1st order) : background, NO conjugate momentum")
    assert len(CONJUGATE_PAIR_LABELS) == 2, "substrate is not 2-sector"
    assert BLOCK_SYMPLECTIC_FORM.shape == (4, 4), "block is not 4x4"
    assert pt.grad_phi is not None  # present, but non-conjugate (background)
    print("     -> PASS: this auxiliary model declares 2 conjugate pairs/node.")
    print("        Its harmonic flow preserves omega and dimension; this test")
    print("        does not certify engine operators or exclude other models.")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(
        "RESULT: the declared auxiliary substrate contains two conjugate\n"
        "  pairs per node\n"
        "  (geometric K_phi/J_phi, potential Phi_s/J_dnfr); |grad phi| is a\n"
        "  non-conjugate background in that model. Its exact harmonic flow is\n"
        "  symplectic and has U(2) polarization symmetry. The associated\n"
        "  oscillator cardinals equal 2, 6, 12, 20 by standard representation\n"
        "  theory; agreement with 2D quantum-dot counts is a correspondence.\n"
        "BASE vs FIBER (the re-located frontier): 2D-ness is a property of\n"
        "  the substrate FIBER (the internal geometric/potential duality,\n"
        "  U(2)). Whether the network base supports an independently derived\n"
        "  effective dimension, and whether any engine operator couples to or\n"
        "  enlarges this auxiliary fiber, remain open."
    )


if __name__ == "__main__":
    main()
