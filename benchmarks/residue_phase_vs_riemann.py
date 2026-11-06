"""Finite directed Paley-spectrum comparison with supplied zeta ordinates.

For the fifteen selected primes p=3 (mod 4), starting at 7, this instrument
constructs the normalized quadratic-residue Cayley operator and compares its
maximum imaginary eigenvalue magnitude with sqrt(p)/(p-1). This is a numerical
check of the classical finite Paley/Gauss-sum formula, not physical phase motion.

The scalar sequence is also paired by list position with the first fifteen
positive zeta-zero ordinates, obtained from mpmath or a supplied rounded table.
Its trend and Pearson correlation test only that declared direct pairing.
A positive correlation would not identify the spectra or support RH; opposite
trends do not exclude transformed or unrelated spectral constructions.
The +0.5 cut is a chosen descriptive threshold, not a theorem of the nodal law.

This source contains fixed comparison choices but no independent freeze/replay
record establishing preregistration. It does not represent analytic S(T) in a
finite automorphism complement or prove a shared obstruction with other models.
See ``theory/TNFR_RIEMANN_RESEARCH_NOTES.md`` sections 5-6. All arithmetic inputs
and comparison ordinates are supplied; no physical or autonomous-emergence
claim follows from the reported numerical match.
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
    print("# DIRECTED PALEY SPECTRUM vs SUPPLIED ZETA ORDINATES — finite comparison")
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
        gamma = [
            14.1347,
            21.0220,
            25.0109,
            30.4249,
            32.9351,
            37.5862,
            40.9187,
            43.3271,
            48.0052,
            49.7738,
            52.9703,
            56.4462,
            59.3470,
            60.8318,
            65.1125,
        ][: len(primes)]

    r_align = float(np.corrcoef(max_im, gamma)[0, 1])
    im_decreasing = all(max_im[i] > max_im[i + 1] for i in range(len(max_im) - 1))
    gamma_increasing = all(gamma[i] < gamma[i + 1] for i in range(len(gamma) - 1))

    print()
    print(f"F-GAUSS  : max|Im| matches sqrt(p)/(p-1) for listed p ?  {f_gauss}")
    print(
        f"F-TREND  : residue content decreasing {im_decreasing} | "
        f"gamma_n increasing {gamma_increasing}  (OPPOSITE)"
    )
    print(
        f"F-ALIGN  : Pearson(max|Im|(p_n), gamma_n) = {r_align:+.4f}  "
        f"(selected descriptive positive-association cut: +0.5)"
    )
    print()

    if f_gauss and r_align < 0.5 and im_decreasing and gamma_increasing:
        verdict = "GAUSS_SCALE_MATCHED_DIRECT_ORDERING_DIFFERS"
    elif r_align > 0.5 and not im_decreasing:
        verdict = "POSITIVE_ASSOCIATION_ONLY"
    else:
        verdict = "INDETERMINATE"
    print(f"VERDICT  : {verdict}")
    print()
    print("Reading: the selected imaginary magnitudes follow the finite Paley")
    print("formula. Trend and correlation concern only this positional pairing")
    print("with supplied zero ordinates; they do not test every transformation")
    print("or establish a representation-theoretic obstruction or an RH result.")
    assert verdict == "GAUSS_SCALE_MATCHED_DIRECT_ORDERING_DIFFERS", verdict


if __name__ == "__main__":
    main()
