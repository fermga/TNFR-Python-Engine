"""TNFR-Riemann P26 demo: Lyapunov-spectral positivity certificate.

Runs :func:`tnfr.riemann.compute_lyapunov_spectral_certificate` on the
P14 prime-ladder Hamiltonian at multiple coupling values and prints:

1. The full certificate at the canonical decoupled limit
   (``coupling=0``).
2. A coupling sweep showing the empirical bottom of the spectrum and
   the Kato-Rellich guaranteed lower bound.
3. An honest-scope reminder: the certificate closes operator-level
   positivity on the finite-dimensional prime-ladder Hilbert space
   but does **not** close gap G4 (RH).

Usage
-----
    set PYTHONPATH=src
    python examples/03_riemann_zeta/53_lyapunov_spectral_positivity_demo.py
"""

from __future__ import annotations

import io
import math
import sys

import numpy as np

# Ensure UTF-8 stdout on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from tnfr.riemann import (
    build_prime_ladder_hamiltonian,
    compute_lyapunov_spectral_certificate,
    compute_spectrum,
    kato_rellich_lower_bound,
)


def banner(title: str) -> None:
    print()
    print("=" * 76)
    print(title)
    print("=" * 76)


def main() -> None:
    n_primes = 12
    max_power = 5

    banner(
        f"P26 :: Decoupled certificate (n_primes={n_primes}, "
        f"max_power={max_power}, J0=0)"
    )
    bundle0 = build_prime_ladder_hamiltonian(
        n_primes, max_power=max_power, coupling=0.0
    )
    cert0 = compute_lyapunov_spectral_certificate(
        bundle0,
        shift=1.0,
        n_unitary_states=10,
        unitary_times=(0.25, 1.0, 4.0),
        seed=20260423,
    )
    print(cert0.summary())
    print()
    print(f"  dimension                : {cert0.dimension}")
    print(
        f"  spectrum (min, max)      : "
        f"({cert0.spectrum_min:.6e}, {cert0.spectrum_max:.6e})"
    )
    print(f"  spectral gap             : {cert0.spectral_gap:.6e}")
    print(f"  all_positive             : {cert0.all_positive}")
    print(f"  ||R||_1 (shift c=1)      : {cert0.schatten_1_norm:.6e}")
    print(f"  ||R||_2 (shift c=1)      : {cert0.schatten_2_norm:.6e}")
    print(f"  unperturbed gap (log p)  : {cert0.unperturbed_gap:.6e}")
    print(f"  perturbation bound ||V|| : {cert0.perturbation_bound:.6e}")
    print(f"  guaranteed gap           : {cert0.guaranteed_gap:.6e}")
    print(f"  perturbation_safe        : {cert0.perturbation_safe}")
    print(f"  max norm drift           : {cert0.max_norm_drift:.3e}")
    print(f"  max energy drift         : {cert0.max_energy_drift:.3e}")
    print(f"  unitary flow certified   : {cert0.unitary}")
    print(f"  STRUCTURAL POSITIVITY    : {cert0.structural_positivity}")

    # ------------------------------------------------------------------
    # Coupling sweep — Kato-Rellich envelope vs empirical spectral bottom
    # ------------------------------------------------------------------
    banner("P26 :: Coupling sweep — Kato-Rellich envelope vs spectrum")
    log2 = math.log(2.0)
    couplings = np.linspace(0.0, 0.30, 7)
    print(
        f"  {'J_0':>8s} | {'min(lambda)':>14s} | "
        f"{'guaranteed_gap':>16s} | {'pert_safe':>10s} | "
        f"{'unitary':>8s}"
    )
    print("  " + "-" * 70)
    for J0 in couplings:
        bundle = build_prime_ladder_hamiltonian(
            n_primes, max_power=max_power, coupling=float(J0)
        )
        spectrum = compute_spectrum(bundle)
        kr = kato_rellich_lower_bound(bundle)
        # Lightweight unitary check (1 state, 1 time) to keep sweep fast
        cert = compute_lyapunov_spectral_certificate(
            bundle,
            shift=max(1.0, -float(spectrum.min()) + 1.0),
            n_unitary_states=2,
            unitary_times=(1.0,),
            seed=20260423,
        )
        print(
            f"  {J0:>8.4f} | "
            f"{float(spectrum.min()):>14.6e} | "
            f"{kr['guaranteed_gap']:>16.6e} | "
            f"{str(kr['perturbation_safe']):>10s} | "
            f"{str(cert.unitary):>8s}"
        )
    print()
    print(f"  Reference: unperturbed gap = log(2) = {log2:.6e}")
    print(
        "  Interpretation: the empirical spectral bottom remains positive\n"
        "  across the whole sweep; the Kato-Rellich envelope guarantees\n"
        "  positivity rigorously while perturbation_bound < log(2)."
    )

    # ------------------------------------------------------------------
    # Honest scope
    # ------------------------------------------------------------------
    banner("Honest scope (do not over-claim)")
    print(
        "  - At J_0 = 0 the certificate is a finite-dimensional restatement\n"
        "    of the trivial fact diag(k log p) > 0; the value is the\n"
        "    explicit gap log 2 plus the Schatten norms used as templates\n"
        "    for the perturbative regime.\n"
        "  - For J_0 > 0 the Kato-Rellich envelope provides a rigorous\n"
        "    quantitative interval in which positivity holds.\n"
        "  - The certificate does NOT prove that non-trivial Riemann zeros\n"
        "    lie on Re(s) = 1/2 (gap G4). It addresses positivity of the\n"
        "    finite-dimensional P14 operator on the prime-ladder space,\n"
        "    which is a necessary but not sufficient ingredient for any\n"
        "    Hilbert-Polya-style attack on RH.\n"
        "  - Pairing with P15 (Weil-Guinand identity) and P16 (Li-Keiper\n"
        "    positivity) provides a stack of operator-level diagnostics;\n"
        "    closing G4 remains genuinely new mathematics."
    )


if __name__ == "__main__":
    main()
