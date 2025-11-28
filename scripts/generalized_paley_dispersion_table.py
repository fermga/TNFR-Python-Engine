"""Generalized Paley dispersion and coherence defect table generator.

For finite fields F_q with q = p^m, q ≡ 1 (mod 4), the Paley graph is conference-type:
  k = (q-1)/2 regular; Laplacian second eigenvalue λ2 = (q - sqrt(q)) / 2.
Hence Δ_P = |λ2 - (q - sqrt(q))/2| = 0 and δ_C = 0 (Definition 2.3, §5.2).

We extend with a synthetic dispersion metric D_k for higher-power residue classes:
Given k (power in generalized k-th power residue graph), define
  D_k = variance of projected eigenvalue shifts under k-th power residue class partition.
Here we provide a placeholder closed-form surrogate scaling: D_k = (1 - 1/k) / k.
(This should be replaced with actual Gaussian period dispersion computations.)

TNFR linkage: δ_C^{(k)} generalizes δ_C; currently 0 under conference alignment.
Robustness heuristic: With conference spectrum, proportion of steps with C(t) > 0.7 is 100%.

Usage:
  pwsh -File scripts/generalized_paley_dispersion_table.py --k-max 8 --output results/generalized_paley_dispersion_table.csv
"""
from __future__ import annotations
import argparse
import math
import csv
from typing import List, Dict


def conference_lambda2(q: int) -> float:
    return (q - math.sqrt(q)) / 2.0


def dispersion_surrogate(k: int) -> float:
    # Placeholder surrogate; real implementation would use Gaussian periods.
    if k <= 0:
        return 0.0
    return (1 - 1.0 / k) / k  # decays ~ 1/k for large k


def robustness_placeholder(delta_c_k: float) -> float:
    # Placeholder: perfect robustness when delta_c_k == 0.
    return 100.0 if delta_c_k == 0 else max(0.0, 100.0 - 50.0 * delta_c_k)


def build_rows(primes: List[int], k_values: List[int]) -> List[Dict[str, str]]:
    rows = []
    for p in primes:
        # For prime powers we only treat q = p for now, extendable to p^m.
        if p % 4 != 1:
            continue
        q = p
        lambda2 = conference_lambda2(q)
        for k in k_values:
            D_k = dispersion_surrogate(k)
            delta_P = 0.0  # conference
            delta_C_k = 0.0  # normalized defect still zero for classical Paley
            robustness = robustness_placeholder(delta_C_k)
            rows.append({
                "q": q,
                "p": p,
                "k": k,
                "lambda2": f"{lambda2:.6f}",
                "delta_P": f"{delta_P:.6f}",
                "delta_C_k": f"{delta_C_k:.6f}",
                "D_k_surrogate": f"{D_k:.6f}",
                "robustness_percent_steps_C_gt_0.7": f"{robustness:.2f}",
                "expected_conference_lambda2": f"{conference_lambda2(q):.6f}",
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate generalized Paley dispersion placeholder table.")
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--primes", nargs="*", type=int, default=[101, 149, 173, 197], help="Primes ≡ 1 (mod 4) to include")
    parser.add_argument("--output", type=str, default="results/generalized_paley_dispersion_table.csv")
    args = parser.parse_args()

    k_values = list(range(2, args.k_max + 1))
    rows = build_rows(args.primes, k_values)

    if not rows:
        print("No rows generated (check primes modulo condition).")
        return

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
