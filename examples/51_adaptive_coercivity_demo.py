"""TNFR-Riemann P24 demo - adaptive sigma refinement for coercivity.

Runs the P22 / P23 empirical interval certificate, then enables P24
adaptive refinement: the worst segments under the segment-local
Lipschitz bound are bisected up to ``refinement_rounds`` times in an
attempt to push ``interval_lb_local`` toward positivity without
inflating claims.
"""

from __future__ import annotations

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from tnfr.riemann import (
    build_prime_ladder_hamiltonian,
    verify_uniform_coercivity_empirical,
)


def main() -> None:
    print("=" * 80)
    print("TNFR-Riemann P24 demo - adaptive coercivity refinement")
    print("=" * 80)

    bundle = build_prime_ladder_hamiltonian(
        n_primes=18,
        max_power=5,
        coupling=0.0,
    )

    cert = verify_uniform_coercivity_empirical(
        bundle,
        sigma_min=0.5,
        sigma_max=4.0,
        n_sigma=10,
        n_zeros=24,
        max_zeros=96,
        refinement_rounds=2,
        refinement_per_round=1,
    )

    print("\n" + cert.summary())

    print("\nInterpretation:")
    print(f"  sampled_all_positive  = {cert.sampled_all_positive}")
    print(f"  admissible_ok         = {cert.admissible_ok}")
    print(f"  nodeaware_ok          = {cert.nodeaware_ok}")
    print(f"  n_sigma (base)        = {cert.n_sigma}")
    print(f"  n_sigma (refined)     = {cert.n_sigma_refined}")
    print(f"  refinement_rounds     = {cert.n_refinement_rounds}")
    print(
        "  interval lb (local)         = "
        f"{cert.interval_lower_bound_local:+.4e}"
    )
    print(
        "  interval lb (local refined) = "
        f"{cert.interval_lower_bound_local_refined:+.4e}"
    )
    print(
        "  interval_lb_local+          = "
        f"{cert.interval_lower_local_positive}"
    )
    print(
        "  interval_lb_local_refined+  = "
        f"{cert.interval_lower_local_refined_positive}"
    )

    delta = (
        cert.interval_lower_bound_local_refined
        - cert.interval_lower_bound_local
    )
    print(f"\n  improvement (refined - local) = {delta:+.4e}")

    print(
        "\nNOTE: adaptive refinement tightens the empirical lower bound "
        "near the bottleneck. Positivity on the refined grid is an "
        "empirical diagnostic, not a proof of uniform coercivity."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
