"""Finite C3 representation and a supplied mass-ratio parametrization.

A three-cycle on the standard K4 representation has cube-root eigenvalues.
This is an algebraic fact about a chosen representation, not a derivation of
particle generations or their relation to primitive TNFR phase.
The script compares embedded mass values with a fitted signed-square-root circle.
For that signed parametrization, the selected sqrt(2) amplitude gives Q=2/3;
using signed roots is not the conventional positive-root relation on every
phase branch. The mean scale, amplitude and fitted phase remain inputs.
Selected static perturbations probe only their stated finite range. They neither
establish a generic failure theorem nor supply a phase or mass evolution law.
Capacity nu_f alone does not imply that every NFR is an oscillator.

Status: auxiliary or finite evidence. See theory/EMERGENT_ONTOLOGY.md and
theory/NODAL_PARAMETER_FOUNDATIONS.md for model and physical-bridge limits.
"""

from __future__ import annotations

import math

import numpy as np


def three_cycle_on_standard_irrep() -> np.ndarray:
    """Eigenvalues of a 3-cycle acting on the standard 3-dim irrep of S_4.

    The tetrahedron's 3-fold internal level is the standard irrep of S_4; a
    3-cycle is a C_3 axis. Restricting the permutation to the sum-zero subspace
    gives its action on the three generations.
    """
    perm = np.array(
        [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=float
    )  # the 3-cycle (1 2 3), fixing vertex 4
    basis = np.array(
        [[1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, -1]], dtype=float
    ).T  # spans the sum-zero (standard-irrep) subspace
    q, _ = np.linalg.qr(basis)  # orthonormal 4x3 basis
    restricted = q.T @ perm @ q
    return np.linalg.eigvals(restricted)


def foot_sqrt_masses(delta: float, scale: float = 1.0) -> np.ndarray:
    """Foot's circle: sqrt(m_k) = M (1 + sqrt2 cos(delta + 2pi k/3)), signed."""
    return np.array(
        [
            scale * (1.0 + math.sqrt(2.0) * math.cos(delta + 2 * math.pi * k / 3))
            for k in range(3)
        ]
    )


def koide(sqrt_masses: np.ndarray) -> float:
    """Koide Q = (sum m) / (sum sqrt m)^2, using the SIGNED square roots."""
    s = np.asarray(sqrt_masses, dtype=float)
    return float((s**2).sum() / (s.sum() ** 2))


def k4_lsym() -> np.ndarray:
    """L_sym of the tetrahedron K_4 (= (4/3)I - (1/3)J); spectrum {0, 4/3 x3}."""
    return (4.0 / 3.0) * np.eye(4) - (1.0 / 3.0) * np.ones((4, 4))


def angle_to_democratic(sqrt_masses: np.ndarray) -> float:
    """Angle (deg) between the sqrt-mass vector and the democratic axis (1,1,1).

    Koide Q = 2/3 <=> this angle is EXACTLY 45 deg: the symmetric (democratic)
    and the symmetry-broken components are equal in magnitude (the sqrt2 = the
    equal-split condition).
    """
    v = np.asarray(sqrt_masses, dtype=float)
    d = np.ones(len(v))
    c = abs(v @ d) / (np.linalg.norm(v) * np.linalg.norm(d))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def main() -> None:
    print("=" * 74)
    print("C3 REPRESENTATION AND A DECLARED MASS-RATIO CIRCLE")
    print("=" * 74)

    # -- M1: the generations are a Z_3 phase-triple (cube roots of unity) ------
    print("\n[M1] Z_3 PHASE-TRIPLE: the C_3 axis acts as the cube roots of unity.")
    ev = three_cycle_on_standard_irrep()
    ev = sorted(ev, key=lambda z: np.angle(z))
    print(f"     {'eigenvalue':>18} {'|.|':>6} {'angle(deg)':>11}")
    for e in ev:
        print(f"     {f'{e:+.3f}':>18} {abs(e):>6.3f} {np.degrees(np.angle(e)):>11.1f}")
    angles = sorted(np.degrees(np.angle(ev)))
    assert all(abs(abs(e) - 1.0) < 1e-9 for e in ev), "not on the unit circle"
    gaps = np.diff(angles + [angles[0] + 360])
    assert all(abs(g - 120.0) < 1e-6 for g in gaps), "not 120 deg apart"
    print("     -> three points at EXACTLY 120 deg on the unit circle: the")
    print("        chosen representation has a cyclic eigenvalue triple.")

    # -- M2: the real leptons ARE on such a circle (Koide 2/3) ------------------
    print("\n[M2] SUPPLIED MASS COMPARISON: fitted circle and Q near 2/3.")
    real = np.array([0.511, 105.66, 1776.86])  # MeV
    q_real = float(real.sum() / (np.sqrt(real).sum() ** 2))
    print(f"     real leptons: Q = {q_real:.5f}   (2/3 = {2/3:.5f})")
    # fit the phase delta that reproduces the real sqrt-mass ratios
    best_d, best_err = 0.0, 1e9
    tgt = np.sort(np.sqrt(real) / np.sqrt(real).min())
    for d in np.linspace(0.0, math.pi / 3, 20000):
        s = np.sort(np.abs(foot_sqrt_masses(d)))
        if s.min() < 1e-6:
            continue
        err = float(np.sum((s / s.min() - tgt) ** 2))
        if err < best_err:
            best_err, best_d = err, d
    s = np.sort(np.abs(foot_sqrt_masses(best_d)))
    print(f"     Foot circle at delta = {math.degrees(best_d):.1f} deg reproduces")
    print(
        f"       sqrt-mass ratios {np.round(s / s.min(), 1)} vs real "
        f"{np.round(tgt, 1)}"
    )
    assert abs(q_real - 2 / 3) < 1e-3, "real leptons not Koide"
    print("     -> the supplied mass values admit the fitted circle comparison.")

    # -- M3: HONEST -- the representation is specified; the circle parameters remain inputs ---------
    print(
        "\n[M3] HONEST BOUNDARY: representation checked, physical identification absent."
    )
    print(f"     {'phase delta':>12} {'Koide Q (signed roots)':>24}")
    for d in (0.1, 0.5, 1.0, 1.9):
        print(f"     {math.degrees(d):>10.0f}   {koide(foot_sqrt_masses(d)):>22.5f}")
    print("     Signed-root Q = 2/3 for this parametrization at every phase --")
    print("     using amplitude sqrt2. The selected tetrahedron perturbations")
    print("     (benchmarks/emergent_generation_count.py M5) gives Q ~ 1/3, NOT 2/3.")
    assert abs(koide(foot_sqrt_masses(0.1)) - 2 / 3) < 1e-9

    # -- M4: TESTED -- does a natural TNFR breaking select sqrt2 (45 deg)? -----
    print("\n[M4] TESTED: can the sqrt2 amplitude EMERGE from a natural breaking?")
    print("     (Koide 2/3 <=> the sqrt-mass vector at EXACTLY 45 deg to (1,1,1).)")
    print(f"     {'natural breaking':>20} {'angle->democ':>13} {'Koide Q':>8}")
    breakings = {
        "C3 axis (vertex)": np.array([0.0, 0.0, 0.0, 1.0]),
        "C2 edge": np.array([0.0, 0.0, 1.0, 1.0]),
        "generic ramp": np.array([0.0, 1.0, 2.0, 3.0]) * 0.3,
    }
    for name, shift in breakings.items():
        excited = np.sort(np.linalg.eigvalsh(k4_lsym() - np.diag(shift)))[1:]
        if excited.min() <= 0:
            print(f"     {name:>20} {'(neg mass)':>13} {'-':>8}")
            continue
        s = np.sqrt(excited)
        print(f"     {name:>20} {angle_to_democratic(s):>13.2f} {koide(s):>8.3f}")
    real_sqrt = np.sqrt(np.array([0.511, 105.66, 1776.86]))
    print(
        f"     {'REAL leptons':>20} {angle_to_democratic(real_sqrt):>13.2f} "
        f"{koide(real_sqrt):>8.3f}"
    )
    assert abs(angle_to_democratic(real_sqrt) - 45.0) < 0.1
    print("     -> each tested perturbation sits NEAR the democratic axis (~2-15")
    print("        deg, Q ~ 1/3, near-degenerate); the REAL leptons are at 45 deg")
    print(
        "        (the MAXIMAL equal-split). These selected perturbations do not give sqrt2."
    )

    print("\n" + "=" * 74)
    print("FINITE COMPARISON:")
    print(
        "  The chosen K4 representation admits a C3 action with cube-root eigenvalues."
    )
    print(
        "  Those eigenvalues are separated by 120 degrees on the complex unit circle."
    )
    print("  This does not identify representation states with particle generations.")
    print("  Embedded mass values are comparison inputs, not TNFR predictions.")
    print(
        "  Their signed-square-root parametrization uses a selected amplitude and phase."
    )
    print("  The finite perturbations do not select the target amplitude in this test.")
    print("  They do not exhaust possible nodal or geometric laws.")
    print("  Q=2/3 follows algebraically for the declared signed-root circle.")
    print(
        "  Positive-root physical interpretation needs its branch and measurement scope."
    )
    print("  A numerical correspondence alone does not derive particle masses.")
    print("  The script supplies no autonomous phase or family-generation mechanism.")
    print("  No physical identification is certified.")
    print("=" * 74)


if __name__ == "__main__":
    main()
