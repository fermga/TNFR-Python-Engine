"""The three generations as a cyclic PHASE-triple: three points at 120 degrees on
a circle (the Koide geometry) -- Layer 3, the pulse/phase reading.

THE INTUITION (theory creator): TNFR is pulse / vibration / phase, not fixed
objects or fixed base parameters. The mass values physics cannot explain
(1 : 207 : 3477) "look like points on a spiral / circle". This is NOT a stray
idea -- it maps onto a real, famous (still unexplained) structure, and TNFR's
own generation structure has exactly that form.

THE REAL STRUCTURE IT MAPS TO (Koide 1981; Foot 1994): the three charged leptons
satisfy the Koide relation Q = (m_e+m_mu+m_tau)/(sqrt m_e+sqrt m_mu+sqrt m_tau)^2
= 2/3 to five digits -- unexplained in ALL of physics. Its geometric reading:
the three sqrt-masses are three points 120 degrees apart on a CIRCLE,
sqrt(m_k) = M (1 + sqrt2 cos(delta + 2pi k/3)). The generations are a cyclic
phase-triple on a circle -- exactly "points on a circle indexed by a phase".

WHY TNFR HAS THIS FORM (derived): the generations are the 3-fold internal level
of the tetrahedron K_4 = the standard irrep of S_4 (benchmarks/emergent_
generation_count.py). A 3-cycle -- a C_3 axis of the tetrahedron -- acts on that
triple with eigenvalues EXACTLY the cube roots of unity {1, e^{+2pi i/3},
e^{-2pi i/3}}: three points 120 degrees apart on the unit circle. So the three
generations are related by a Z_3 PHASE -- a cyclic phase-triple, not three
independent objects. That is the pulse/phase ontology (AGENTS.md: every NFR a
phase oscillator), and it is the SAME 120-on-a-circle form the Koide geometry
needs.

WHAT EMERGES (measured):
  - M1: the C_3 axis of the tetrahedron acts on the 3 generations with
    eigenvalues = the cube roots of unity (120 deg apart, unit circle). The
    generations are a Z_3 phase-triple -- points on a circle indexed by a phase.
  - M2: the real leptons ARE three points on such a circle -- Koide Q = 2/3
    (verified), and the Foot parametrization reproduces the real masses at a
    phase delta ~ 12 deg. "Points on a circle" is literally the real structure.
  - M3 (HONEST): the phase circle gives Koide Q = 2/3 for EVERY phase delta (a
    structural identity of three points at 120 deg) -- but only with the specific
    amplitude sqrt2. A GENERIC TNFR tetrahedron breaking gives Q ~ 1/3, NOT 2/3.
    So TNFR derives the FORM (120-deg cyclic phase-triple) but not the specific
    circle (the sqrt2 amplitude, the phase delta) that fixes Koide and the masses.

THE HONEST VERDICT: the intuition is RIGHT about the FORM and WRONG about being a
throwaway -- the generations are a cyclic phase-triple (three points at 120 deg on
a circle), derived from the S_4 / C_3 structure, and this is exactly the Koide
geometry. What is NOT derived is the specific circle: the sqrt2 amplitude (which
makes Q = 2/3) and the phase delta (which fixes 1 : 207 : 3477). Geometrically
sqrt2 <=> the sqrt-mass vector at EXACTLY 45 deg to the democratic axis (1,1,1),
i.e. the symmetric and symmetry-broken parts equal in magnitude -- the MAXIMAL
equal-split. M4 TESTS whether a natural TNFR breaking selects it: it does NOT --
every natural breaking (C_3 axis, C_2 edge, generic ramp) sits near the democratic
axis (~2-15 deg, Q ~ 1/3, near-degenerate), the OPPOSITE extreme from the real
leptons' 45 deg. So sqrt2 does NOT emerge from generic dynamics; claiming it does
would be numerology. Koide's 2/3 is itself unexplained in all of physics -- the
leptons sit at a special (maximal equal-split) point generic dynamics does not
reach. "Spiral" (rather than a single circle) is the natural cross-FAMILY
extension -- a growing radius from leptons to quarks -- noted as a direction.

HONEST SCOPE: M1 is exact S_4 representation theory (the cube-root-of-unity action
of a 3-cycle on the standard irrep) -- standard mathematics, the TNFR content
being that the generations ARE this triple (emergent_generation_count.py). M2 is
the empirical Koide relation + Foot's parametrization (real physics, not TNFR).
M3-M4 are the honest boundary: the sqrt2 (45-deg equal-split) and the phase delta
are TESTED and do NOT emerge from natural breakings. Nothing here derives the
lepton masses or closes Koide; it identifies the FORM (a cyclic phase-triple on a
circle) that TNFR supplies and the specific circle it does not. Closes no problem.

Run:
    python benchmarks/emergent_generation_phase_circle.py

Theoretical anchor: AGENTS.md (the pulse; every NFR a phase oscillator; phase
channel); theory/EMERGENT_ONTOLOGY.md Sec.9.1 (OPEN properties); benchmarks/
emergent_generation_count.py (the 3 generations = the S_4 standard irrep).
Status: RESEARCH (Layer-3 phase-geometry reading; honest form-vs-values split).
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
    print("THE THREE GENERATIONS AS A CYCLIC PHASE-TRIPLE (points on a circle)")
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
    print("        generations are a cyclic phase-triple, not 3 fixed objects.")

    # -- M2: the real leptons ARE on such a circle (Koide 2/3) ------------------
    print("\n[M2] THE REAL LEPTONS ON A CIRCLE: Koide Q = 2/3 (Foot's geometry).")
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
    print("     -> the real generations ARE three points at 120 deg on a circle.")

    # -- M3: HONEST -- the FORM is derived, the specific circle is not ---------
    print("\n[M3] HONEST BOUNDARY: form derived, specific circle NOT.")
    print(f"     {'phase delta':>12} {'Koide Q (signed roots)':>24}")
    for d in (0.1, 0.5, 1.0, 1.9):
        print(f"     {math.degrees(d):>10.0f}   {koide(foot_sqrt_masses(d)):>22.5f}")
    print("     Koide Q = 2/3 for EVERY phase (a 120-deg-triple identity) -- but")
    print("     ONLY with the amplitude sqrt2. A GENERIC tetrahedron breaking")
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
    print("     -> every natural breaking sits NEAR the democratic axis (~2-15")
    print("        deg, Q ~ 1/3, near-degenerate); the REAL leptons are at 45 deg")
    print("        (the MAXIMAL equal-split). No natural TNFR breaking gives sqrt2.")

    print("\n" + "=" * 74)
    print("VERDICT (Layer 3, the phase reading):")
    print("  DERIVED (the FORM): the 3 generations are a Z_3 phase-triple -- three")
    print("    points at 120 deg on a circle (the C_3 action on the S_4 triplet).")
    print("    This is exactly the intuition, and exactly the Koide geometry.")
    print("  REAL PHYSICS: the leptons sit on that circle (Koide Q = 2/3), at 45")
    print("    deg to the democratic axis -- the MAXIMAL equal-split point.")
    print("  TESTED, NOT DERIVED: no natural TNFR breaking selects sqrt2 -- generic")
    print("    breakings give ~1/3 (near-degenerate), the OPPOSITE extreme. The")
    print("    sqrt2 (45 deg equal-split) and the phase delta are NOT emergent here;")
    print("    claiming otherwise would be numerology. Koide's 2/3 is unexplained")
    print("    in ALL physics -- the leptons sit at a special point generic dynamics")
    print("    does not reach. 'Spiral' = the cross-family extension, still a")
    print("    direction. The FORM emerges; the two parameters do NOT.")
    print("=" * 74)


if __name__ == "__main__":
    main()
