"""Wave-particle correspondence: the wave's own mode index IS the particle's
topological charge (the classical, ring-topology instance of the textbook
"particle on a ring" quantization).

THE QUESTION (theory creator): from the emergent structural cosmology (Sec.2.5),
could wave-particle duality be understood structurally? The stage (Sec.7.1a, a
discrete standing-mode spectrum, wave-like: omega_k=sqrt(lambda_k)) and the
occupant (Sec.7.1b, a conserved integer topological winding W, particle-like)
were derived SEPARATELY. This asks whether they are two readings of the SAME
object.

THE IDENTITY (derived here): on a ring, the k-th stage eigenmode of L_sym, read
as the COMPLEX order parameter Psi = e^(i*theta) (the same complex field used
throughout Sec.7 for the winding), is an EXACT eigenvector of L_sym with the
canonical eigenvalue lambda_k = 1 - cos(2 pi k / n) -- AND it carries topological
winding EXACTLY k. So the wave's own mode index k (its spatial frequency, its
dispersion omega_k = sqrt(lambda_k)) IS the particle's winding charge W = k: not
a duality/trade-off, but a literal identity of the two labels, on the ring. This
is the classical-field-theory content that underlies the textbook "particle on a
ring" result in quantum mechanics (single-valuedness of e^(i*k*theta) around a
loop forces k in Z, and k is simultaneously the wavefunction's spatial frequency
and the particle's quantized angular momentum) -- TNFR's nodal operator contains
the SAME topological fact.

THE COMPLEMENT (also derived here): a REAL phase ripple built from a few low-k
stage modes (a generic "wave" excitation of the vacuum, e.g. cos(2 pi k i / n))
stays at winding 0 for ANY amplitude tested (up to 20 rad, far past any small-
perturbation regime) -- the winding charge is NOT reached by growing a narrow-
band real wave; it requires the SPECIFIC complex k-th mode of THE IDENTITY above.
So "wave" (a real ripple around the vacuum) and "particle" (a winding defect) are
topologically distinct sectors of the SAME field, connected only through the
complex order parameter's own mode structure.

WHAT EMERGES (measured):
  - M1: the k-th complex stage mode is an EXACT L_sym eigenvector (residual
    ~1e-15) with the canonical eigenvalue, AND its topological winding is EXACTLY
    k -- the wave's mode index and the particle's charge are the SAME integer.
  - M2: real narrow-band ripples (a few low-k modes) stay at winding 0 for every
    amplitude tested (0.5 to 20 rad) -- generic wave excitations of the vacuum
    are topologically trivial; they do not "leak" into a charged sector.

HONEST SCOPE: this is the standard topological fact behind single-valued maps to
a circle (winding number, homotopy classes of S^1 -> S^1) applied to TNFR's own
complex order parameter Psi = K_phi + i J_phi (Sec.5.3, Sec.7.1b) -- the same
classical mathematics that underlies the quantum "particle on a ring" angular-
momentum quantization, re-expressed on the canonical structural operator. It is a
DERIVED structural correspondence between the already-derived stage (Sec.7.1a)
and occupant (Sec.7.1b). It is NOT the full quantum-mechanical wave-particle
duality: there is no probability amplitude, no Born rule, no single-particle
interference statistics, and no de Broglie relation with a physical hbar -- the
substrate is classical (Sec.9.2, OPEN). Closes no open problem.

Run:
    python benchmarks/emergent_wave_particle_correspondence.py

Theoretical anchor: theory/EMERGENT_ONTOLOGY.md Sec.7.1 (stage / occupant),
Sec.5.3 (the complex geometric field Psi), Sec.9.2 (the classical-substrate
boundary); benchmarks/emergent_particle_catalog.py (winding_ring, winding_number).
Status: RESEARCH (structural correspondence; stage=occupant identity on the ring).
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np

_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tnfr.physics.emergent_particles import (  # noqa: E402
    winding_number,
    winding_ring,
)
from tnfr.physics.structural_diffusion import (  # noqa: E402
    symmetric_normalized_laplacian,
)


def main() -> None:
    print("=" * 74)
    print("WAVE-PARTICLE CORRESPONDENCE: the mode index IS the winding charge")
    print("=" * 74)

    n = 60
    _, lsym = symmetric_normalized_laplacian(winding_ring(n, 0))
    lsym = np.asarray(lsym)

    # -- M1: the k-th complex stage mode IS an L_sym eigenvector, AND its ------
    # -- winding is exactly k (wave-mode-index = particle-charge, exact) -------
    print("\n[M1] the k-th complex stage mode: eigenvector + winding = k (exact).")
    print(
        f"     {'k':>3} {'lambda_k (meas)':>16} {'1-cos(2pi k/n)':>16} "
        f"{'residual':>10} {'winding':>8}"
    )
    for k in (1, 2, 3, 5, 10):
        psi = np.exp(1j * 2 * math.pi * k * np.arange(n) / n)
        l_psi = lsym @ psi
        lam_meas = float(np.real(np.vdot(psi, l_psi) / np.vdot(psi, psi)))
        residual = float(np.max(np.abs(l_psi - lam_meas * psi)))
        lam_pred = 1.0 - math.cos(2 * math.pi * k / n)
        w, raw = winding_number(winding_ring(n, k))
        print(
            f"     {k:>3} {lam_meas:>16.6f} {lam_pred:>16.6f} "
            f"{residual:>10.2e} {w:>8d}"
        )
        assert residual < 1e-9, f"mode {k} is not an eigenvector"
        assert abs(lam_meas - lam_pred) < 1e-9, f"eigenvalue mismatch at k={k}"
        assert w == k and abs(raw - k) < 1e-9, f"winding != k at k={k}"
    print("     -> PASS: the wave's OWN mode index (its dispersion omega_k=")
    print("        sqrt(lambda_k)) IS the particle's topological charge W=k.")

    # -- M2: real narrow-band ripples stay at winding 0 (topologically trivial)-
    print("\n[M2] real narrow-band ripples (low-k modes): winding vs amplitude.")
    rng = np.random.default_rng(0)
    ks = (1, 2, 3)
    coeffs = rng.uniform(-1, 1, size=len(ks))
    idx = np.arange(n)
    print(f"     {'amplitude':>10} {'winding':>8}")
    for amp in (0.5, 2.0, 5.0, 10.0, 20.0):
        theta = np.zeros(n)
        for c, k in zip(coeffs, ks):
            theta += amp * c * np.cos(2 * math.pi * k * idx / n)
        theta = np.mod(theta, 2 * math.pi)
        G = winding_ring(n, 0)
        for i in range(n):
            G.nodes[i]["theta"] = float(theta[i])
            G.nodes[i]["phase"] = float(theta[i])
        w, _ = winding_number(G)
        print(f"     {amp:>10.1f} {w:>8d}")
        assert w == 0, f"a narrow-band real ripple acquired charge at amp={amp}"
    print("     -> PASS: a generic real 'wave' excitation of the vacuum stays")
    print("        topologically trivial (W=0) at every amplitude tested.")

    print("\n" + "=" * 74)
    print("THE CORRESPONDENCE:")
    print("  wave (stage, Sec.7.1a): mode index k, dispersion omega_k=sqrt(lambda_k)")
    print("  particle (occupant, Sec.7.1b): topological charge W")
    print("  on the ring: W = k, EXACTLY -- the same integer labels both readings.")
    print("  A real narrow-band ripple stays W=0: 'wave' and 'particle' are")
    print("  distinct sectors of the same field, joined by the complex mode.")
    print("HONEST: the classical topology of maps S^1->S^1 (the textbook 'particle")
    print("  on a ring' angular-momentum quantization); no probability amplitude,")
    print("  Born rule, interference statistics or physical hbar (Sec.9.2, OPEN).")
    print("=" * 74)


if __name__ == "__main__":
    main()
