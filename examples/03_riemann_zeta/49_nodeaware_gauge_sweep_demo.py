"""TNFR-Riemann P20 demo - node-aware gauge sweep for alpha.

Sweeps alpha = W / E_TNFR over:
- sigma grid
- admissible test families (P19)
- node-aware gauges depending on nu_f and node weight

Goal: probe robustness when gauges explicitly depend on structural
frequency and node-level weighting channels.
"""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from tnfr.riemann import build_prime_ladder_hamiltonian, sweep_alpha_nodeaware


def main() -> None:
    print("=" * 80)
    print("TNFR-Riemann P20 demo - node-aware gauge alpha sweep")
    print("=" * 80)

    n_primes = 18
    max_power = 5
    dim = n_primes * max_power

    bundle = build_prime_ladder_hamiltonian(
        n_primes=n_primes,
        max_power=max_power,
        coupling=0.0,
    )
    print(f"\nBundle: n_primes={n_primes}, " f"max_power={max_power}, dim={dim}")

    sigmas = np.logspace(np.log10(0.5), np.log10(8.0), 10).tolist()
    print("Sigma grid:")
    print("  " + "  ".join(f"{s:.3f}" for s in sigmas))

    cert = sweep_alpha_nodeaware(
        bundle,
        sigmas,
        n_zeros=40,
        max_zeros=160,
    )

    print("\n" + cert.summary())

    print("\nFamilies:", ", ".join(cert.families))
    print("Node-aware gauges:", ", ".join(cert.node_gauges))

    print("\nalpha_min per family (across node-aware gauges and sigmas):")
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
        f"node_gauge='{cert.alpha_min_node_gauge}')"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
