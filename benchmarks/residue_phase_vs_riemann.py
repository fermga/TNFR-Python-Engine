"""
benchmarks/residue_phase_vs_riemann.py

PRE-REGISTERED FALSIFIER (R-infinity discipline) — the "refactor the Riemann
attack" hypothesis from the number-theory reframe (TNFR_NUMBER_THEORY.md §11).

QUESTION. The TNFR-Riemann program is paused at T-HP on the SELF-ADJOINT
prime-ladder P14, walled by the Euler-Orthogonality Lemma (everything commutes
with the S_n prime-relabelling -> spectrum in Fix(S_n), blind to
S(T) = (1/pi) arg zeta(1/2+iT) in Fix(S_n)^perp). The number-theory reframe
(§9.6) supplies a DIFFERENT object: the directed quadratic-residue diffusion
operator L_rw on the Paley tournament (n = 3 mod 4), which is NON-NORMAL, has a
COMPLEX spectrum, and carries the arithmetic in the PHASE (imaginary part) -- and
whose symmetry is the affine group of Z/n, NOT S_n. Does this non-normal phase
operator carry the Riemann zeros {gamma_n}, or only Gauss-sum / sqrt(p) content?

CONSTRUCTION (deterministic; only arithmetic input is x^2 mod p).
  For primes p = 3 (mod 4): G = arithmetic_cayley_digraph(p, QR(p));
  L_rw = structural_diffusion_operator(G) (the canonical dNFR EPI channel);
  take its complex spectrum and the phase content max|Im(lambda)|.

PRE-REGISTERED METRICS + THRESHOLDS (locked before reading results).
  F-GAUSS    : max|Im(lambda)|(p) == sqrt(p)/(p-1) for all p (|ratio-1| < 1e-9).
               [identifies the phase content as the Paley Gauss-sum eigenvalue]
  F-TREND    : sign of the trend of max|Im|(p_n) vs n, compared to gamma_n.
               Riemann gamma_n strictly INCREASE; Gauss content ~1/sqrt(p)
               strictly DECREASES.
  F-ALIGN    : Pearson(max|Im|(p_n), gamma_n) over the n-th 3-mod-4 prime and the
               n-th zero. Aligned (Riemann) would need r > +0.5.

PRE-REGISTERED VERDICT LOGIC.
  GAUSS_CONFIRMED_RIEMANN_REFUTED : F-GAUSS holds AND F-ALIGN r < +0.5 AND the
        residue content decreases while gamma_n increase.
  RIEMANN_SUPPORTED               : F-ALIGN r > +0.5 with matching growth.
  INDETERMINATE                   : otherwise.

HONEST PREDICTION (declared in advance): GAUSS_CONFIRMED_RIEMANN_REFUTED. The
non-normal phase operator reaches arithmetic in the phase -- a genuine advance
over the self-adjoint mod-4 restriction, and it EVADES the Euler-Orthogonality
Lemma -- but the phase content is sqrt(p) Gauss sums, not {gamma_n}. The
residue-phase -> zeta-zeros bridge is the SAME e-pi / Fix(S_n)^perp wall
(§9.5 "both walls coincide"). This sharpens the wall; it does not dissolve it.
Closes NO open problem; the TNFR-Riemann program stays paused at T-HP, G4=RH OPEN.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tnfr.mathematics.number_theory import (  # noqa: E402
    arithmetic_cayley_digraph,
    quadratic_residue_set,
)
from tnfr.physics.structural_diffusion import (  # noqa: E402
    structural_diffusion_operator,
)

try:
    import mpmath as mp

    _HAVE_MP = True
except ImportError:
    _HAVE_MP = False


def _operator_matrix(G):
    ret = structural_diffusion_operator(G)
    for obj in ret:
        arr = np.asarray(obj)
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            return arr.astype(complex)
    raise RuntimeError("no square operator matrix returned")


def _is_prime(n: int) -> bool:
    return n >= 2 and all(n % d for d in range(2, int(n**0.5) + 1))


def _primes_3mod4(count: int) -> list[int]:
    out, n = [], 7
    while len(out) < count:
        if n % 4 == 3 and _is_prime(n):
            out.append(n)
        n += 2
    return out


def main() -> None:
    print("#" * 72)
    print("# RESIDUE PHASE vs RIEMANN — pre-registered falsifier")
    print("#" * 72)
    print()

    primes = _primes_3mod4(15)
    max_im, gauss_pred = [], []
    print(" p (3mod4)   max|Im(lambda)|   sqrt(p)/(p-1)    ratio")
    f_gauss = True
    for p in primes:
        op = _operator_matrix(arithmetic_cayley_digraph(p, quadratic_residue_set(p)))
        eig = np.linalg.eigvals(op)
        mi = float(np.max(np.abs(eig.imag)))
        gp = math.sqrt(p) / (p - 1)
        ratio = mi / gp
        f_gauss &= abs(ratio - 1.0) < 1e-9
        max_im.append(mi)
        gauss_pred.append(gp)
        print(f"  {p:4d}       {mi:.6f}         {gp:.6f}      {ratio:.6f}")

    # Riemann zeros gamma_n (oracle, mpmath) for the alignment test
    if _HAVE_MP:
        gamma = [float(mp.zetazero(n + 1).imag) for n in range(len(primes))]
    else:
        gamma = [14.1347, 21.0220, 25.0109, 30.4249, 32.9351, 37.5862, 40.9187,
                 43.3271, 48.0052, 49.7738, 52.9703, 56.4462, 59.3470, 60.8318,
                 65.1125][: len(primes)]

    r_align = float(np.corrcoef(max_im, gamma)[0, 1])
    im_decreasing = all(max_im[i] > max_im[i + 1] for i in range(len(max_im) - 1))
    gamma_increasing = all(gamma[i] < gamma[i + 1] for i in range(len(gamma) - 1))

    print()
    print(f"F-GAUSS  : max|Im| == sqrt(p)/(p-1) for all p ?  {f_gauss}")
    print(f"F-TREND  : residue content decreasing {im_decreasing} | "
          f"gamma_n increasing {gamma_increasing}  (OPPOSITE)")
    print(f"F-ALIGN  : Pearson(max|Im|(p_n), gamma_n) = {r_align:+.4f}  "
          f"(need > +0.5 for Riemann)")
    print()

    if f_gauss and r_align < 0.5 and im_decreasing and gamma_increasing:
        verdict = "GAUSS_CONFIRMED_RIEMANN_REFUTED"
    elif r_align > 0.5 and not im_decreasing:
        verdict = "RIEMANN_SUPPORTED"
    else:
        verdict = "INDETERMINATE"
    print(f"VERDICT  : {verdict}")
    print()
    print("Reading: the non-normal phase operator EVADES the Euler-Orthogonality")
    print("wall (affine symmetry, not S_n) and reaches arithmetic in the phase --")
    print("but the content is sqrt(p) Gauss sums (decreasing ~1/sqrt(p)), NOT the")
    print("zeta zeros (increasing). The residue-phase -> zeta-zeros bridge is the")
    print("same e-pi / Fix(S_n)^perp wall. Sharpens the wall; closes nothing.")
    assert verdict == "GAUSS_CONFIRMED_RIEMANN_REFUTED", verdict


if __name__ == "__main__":
    main()
