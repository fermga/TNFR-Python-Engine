"""Paley gap and coherence defect table generator.

Computes λ2 (second smallest Laplacian eigenvalue), Paley gap Δ_P = |λ2 - (n - sqrt(n))/2|,
normalized coherence defect δ_C = Δ_P / sqrt(n) for residue circulant graphs over Z_n
with n ≡ 1 (mod 4). Distinguishes prime vs composite factorization patterns.

TNFR linkage: δ_C = 0 corresponds to conference spectral alignment (Interpretation 5.2);
δ_C > 0 indicates coherence defect induced by zero divisors.

Usage (PowerShell):
  pwsh -File scripts/paley_gap_table.py --n-min 101 --n-max 1000 --output results/paley_gap_table.csv --only-1mod4

To specify explicit values:
  pwsh -File scripts/paley_gap_table.py --values 101 149 173 25777

"""
from __future__ import annotations
import argparse
import math
import csv
from typing import List, Tuple
import numpy as np

try:
    import sympy as sp
    HAVE_SYMPY = True
except ImportError:  # minimal fallback primality test
    HAVE_SYMPY = False


def is_prime(n: int) -> bool:
    if HAVE_SYMPY:
        return bool(sp.isprime(n))
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    r = int(math.isqrt(n))
    f = 3
    while f <= r:
        if n % f == 0:
            return False
        f += 2
    return True


def quadratic_residues_mod_n(n: int) -> set:
    # Non-zero residues only (Paley-style); zero excluded to avoid self-loops.
    residues = set()
    for x in range(1, n):
        residues.add((x * x) % n)
    residues.discard(0)
    return residues


def build_residue_graph_adjacency(n: int) -> np.ndarray:
    residues = quadratic_residues_mod_n(n)
    # Build adjacency via broadcasting differences; symmetric condition: diff in residues OR -diff in residues.
    indices = np.arange(n)
    diff = (indices[:, None] - indices[None, :]) % n  # shape (n,n)
    mask = np.vectorize(lambda d: (d in residues) or ((-d) % n in residues))(diff)
    np.fill_diagonal(mask, False)
    return mask.astype(np.int8)


def laplacian_second_eigenvalue(adj: np.ndarray) -> float:
    degrees = adj.sum(axis=1)
    L = np.diag(degrees) - adj
    # Use eigh for symmetric real matrix; Laplacian is symmetric.
    evals = np.linalg.eigvalsh(L)
    # Numerical stability: sort ascending; λ0 ≈ 0.
    evals.sort()
    return float(evals[1])


def compute_metrics(n: int) -> Tuple[int, str, float, float, float, int]:
    adj = build_residue_graph_adjacency(n)
    lambda2 = laplacian_second_eigenvalue(adj)
    target = (n - math.sqrt(n)) / 2.0
    delta_p = abs(lambda2 - target)
    delta_c = delta_p / math.sqrt(n)
    classification = "prime" if is_prime(n) else "composite"
    distinct = len({*(np.linalg.eigvalsh(np.diag(adj.sum(axis=1)) - adj))})
    return n, classification, lambda2, delta_p, delta_c, distinct


def generate_values(args) -> List[int]:
    if args.values:
        return sorted(set(args.values))
    values = []
    for n in range(args.n_min, args.n_max + 1):
        if args.only_1mod4 and n % 4 != 1:
            continue
        values.append(n)
    return values


def main():
    parser = argparse.ArgumentParser(description="Generate Paley gap table for residue circulant graphs over Z_n.")
    parser.add_argument("--n-min", type=int, default=101)
    parser.add_argument("--n-max", type=int, default=1000)
    parser.add_argument("--values", nargs="*", type=int, help="Explicit n values")
    parser.add_argument("--only-1mod4", action="store_true", help="Restrict to n ≡ 1 (mod 4)")
    parser.add_argument("--output", type=str, default="results/paley_gap_table.csv")
    parser.add_argument("--max-n", type=int, default=2000, help="Safety upper bound to prevent accidental huge matrices")
    args = parser.parse_args()

    if args.n_max > args.max_n:
        raise ValueError(f"n_max {args.n_max} exceeds safety max_n {args.max_n}")

    values = generate_values(args)
    rows = []
    for n in values:
        n_val, cls, lambda2, delta_p, delta_c, distinct = compute_metrics(n)
        rows.append({
            "n": n_val,
            "type": cls,
            "lambda2": f"{lambda2:.6f}",
            "delta_P": f"{delta_p:.6f}",
            "delta_C": f"{delta_c:.6f}",
            "distinct_L_eigs": distinct,
            "expected_conference_lambda2": f"{(n - math.sqrt(n))/2.0:.6f}",
        })

    out_path = args.output
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
