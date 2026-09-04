#!/usr/bin/env python3
"""R6 additive-resonance controlled benchmark.

Shows (a) the exact reduction r_2 = IFFT(hat 1_P^2) — the additive-character
reading of Goldbach IS the Fourier / circle-method description; and (b) the
constructive negative: the candidate TNFR phase observable (phase-curvature
energy) does NOT exceed the classical power-spectrum discrimination of the primes
under density-matched controls (matched, Cramer, shuffle).

Honest scope: no proof of Goldbach or any open problem; the target property is
never used to define the phase.  The claim "TNFR additive phase adds information
beyond classical Fourier" (NT-P06) is OPEN and, for the observable tested here,
NEGATIVE.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from tnfr.mathematics.additive_resonance import (  # noqa: E402
    excess_over_fourier,
    goldbach_fourier_residual,
    prime_indicator,
)
from tnfr.research import (  # noqa: E402
    CircularityAudit,
    ClaimStatus,
    ExperimentManifest,
    input_bit_length,
)

SIZES = [128, 256, 512, 1024]
CONTROL_KINDS = ["matched", "cramer", "shuffle"]


def main() -> int:
    print("R6A exact reduction: r_2 = IFFT(hat 1_P^2) (additive reading = Fourier)")
    reduction_ok = True
    for n in SIZES:
        ind = prime_indicator(n)
        resid = goldbach_fourier_residual(ind)
        tol = np.finfo(float).eps * n * max(float(ind.sum()), 1.0) ** 2
        ok = resid <= tol
        reduction_ok &= ok
        print(f"  N={n:>4}  residual={resid:.2e}  tol={tol:.2e}  {ok}")

    print()
    print("R6B constructive negative: TNFR phase vs classical Fourier (z-scores)")
    print(f"  {'control':>8} {'classical_z':>12} {'tnfr_z':>8} {'exceeds':>8}")
    never_exceeds = True
    for kind in CONTROL_KINDS:
        r = excess_over_fourier(1024, seed=0, n_controls=30, kind=kind)
        never_exceeds &= (not r.tnfr_exceeds)
        print(f"  {kind:>8} {r.classical_z:>12.2f} {r.tnfr_z:>8.2f} "
              f"{str(r.tnfr_exceeds):>8}")

    audit = CircularityAudit()  # phase defined without the target property
    manifest = ExperimentManifest(
        claim_id="NT-P06",
        git_sha="local",
        versions={"python": platform.python_version(),
                  "numpy": np.__version__},
        seed=0,
        operator_sequence=(),
        uses_known_factors=False,
        input_bits=input_bit_length(max(SIZES)),
        controls=("matched_random", "cramer", "shuffle",
                  "prime_constellation", "phase_convention"),
        artifacts=(),
    )
    print()
    print(f"  exact reduction holds (all sizes): {reduction_ok}")
    print(f"  TNFR never exceeds Fourier (all)  : {never_exceeds}")
    print(f"  reduction claim : {ClaimStatus.DERIVED.value} + measured")
    print(f"  excess claim    : {ClaimStatus.NEGATIVE.value} "
          "(NT-P06 OPEN; tested observable shows no excess)")
    print(f"  circularity     : {audit.verdict.value}")
    print(f"  input bits (max): {manifest.input_bits}")
    return 0 if (reduction_ok and never_exceeds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
