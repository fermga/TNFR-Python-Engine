"""TNFR-Riemann P22 demo - empirical uniform coercivity certificate.

Builds an interval-level coercivity diagnostic by combining P19 and P20
alpha surfaces and estimating a mesh-corrected lower bound.
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
    print("TNFR-Riemann P22 demo - empirical uniform coercivity")
    print("=" * 80)

    bundle = build_prime_ladder_hamiltonian(
        n_primes=18,
        max_power=5,
        coupling=0.0,
    )

    cert = verify_uniform_coercivity_empirical(
        bundle,
        sigma_min=0.5,
        sigma_max=8.0,
        n_sigma=12,
        n_zeros=30,
        max_zeros=120,
    )

    print("\n" + cert.summary())

    print("\nInterpretation:")
    print(f"  sampled_all_positive  = {cert.sampled_all_positive}")
    print(f"  admissible_ok         = {cert.admissible_ok}")
    print(f"  nodeaware_ok          = {cert.nodeaware_ok}")
    print(f"  interval_lb_positive  = {cert.interval_lower_positive}")
    print(
        "  interval lower bound  = "
        f"{cert.interval_lower_bound:+.4e}"
    )
    print(
        "\nNOTE: this is an empirical interval certificate, not a "
        "full analytic proof of uniform coercivity."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
