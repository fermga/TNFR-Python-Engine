"""TNFR-Riemann P19 demo - admissible-family sweep for alpha.

Runs alpha = W / E_TNFR over:
- sigma grid (log-spaced)
- default test families (gaussian, gaussian_mixture, hermite2_gaussian)
- default structural gauges from P18

Goal: stress-test positivity beyond a single Gaussian family.
"""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from tnfr.riemann import build_prime_ladder_hamiltonian, sweep_alpha_admissible_family


def main() -> None:
    print("=" * 78)
    print("TNFR-Riemann P19 demo - admissible-family alpha sweep")
    print("=" * 78)

    n_primes = 18
    max_power = 5
    bundle = build_prime_ladder_hamiltonian(
        n_primes=n_primes,
        max_power=max_power,
        coupling=0.0,
    )
    dim = n_primes * max_power
    print(f"\nBundle: n_primes={n_primes}, " f"max_power={max_power}, dim={dim}")

    sigmas = np.logspace(np.log10(0.5), np.log10(8.0), 10).tolist()
    print("Sigma grid:")
    print("  " + "  ".join(f"{s:.3f}" for s in sigmas))

    cert = sweep_alpha_admissible_family(
        bundle,
        sigmas,
        n_zeros=40,
        max_zeros=160,
    )

    print("\n" + cert.summary())

    print("\nFamilies:", ", ".join(cert.families))
    print("Gauges:", ", ".join(cert.gauges))

    print("\nW[sigma] by family (gauge-independent):")
    for i, fam in enumerate(cert.families):
        vals = "  ".join(f"{float(v):+.3e}" for v in cert.weil_table[i])
        print(f"  {fam:>18s}: {vals}")

    print("\nalpha_min per family (across gauges and sigmas):")
    for i, fam in enumerate(cert.families):
        alpha_slice = cert.alpha_table[i]
        finite = alpha_slice[np.isfinite(alpha_slice)]
        if finite.size > 0:
            amin = float(finite.min())
            amax = float(finite.max())
            print(f"  {fam:>18s}: min={amin:+.4e}, max={amax:+.4e}")
        else:
            print(f"  {fam:>18s}: no finite alpha values")

    print("\nInterpretation:")
    print(f"  W_all_positive     = {cert.weil_all_positive}")
    print(f"  alpha_all_positive = {cert.alpha_all_positive}")
    print(
        "  Tightest triple    = "
        f"(sigma={cert.alpha_min_sigma:.3f}, "
        f"family='{cert.alpha_min_family}', "
        f"gauge='{cert.alpha_min_gauge}')"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
