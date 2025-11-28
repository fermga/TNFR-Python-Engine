#!/usr/bin/env python3
"""
TNFR primality checker (equations-only)

Usage:
  python scripts/tnfr_is_prime.py 17 99991 999983

Determines primality via the TNFR arithmetic pressure equation:
  ΔNFR(n) = ζ·(ω(n)−1) + η·(τ(n)−2) + θ·(σ(n)/n − (1+1/n))
A number is prime iff ΔNFR(n) == 0.

No factorization beyond what is necessary to compute τ, σ, ω is used.
"""
from __future__ import annotations

import argparse
from typing import Tuple, List

# Direct arithmetic helpers (trial-division based)


def divisor_count(n: int) -> int:
    cnt = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            cnt += 1
            if i != n // i:
                cnt += 1
        i += 1
    return cnt


def divisor_sum(n: int) -> int:
    total = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            total += i
            j = n // i
            if j != i:
                total += j
        i += 1
    return total


def prime_factor_count(n: int) -> int:
    count = 0
    d = 2
    while d * d <= n:
        while n % d == 0:
            count += 1
            n //= d
        d += 1
    if n > 1:
        count += 1
    return count


def tnfr_delta_nfr(n: int, *, zeta=1.0, eta=0.8, theta=0.6) -> float:
    if n < 2:
        return float('inf')
    tau_n = divisor_count(n)
    sigma_n = divisor_sum(n)
    omega_n = prime_factor_count(n)
    factorization_pressure = zeta * (omega_n - 1)
    divisor_pressure = eta * (tau_n - 2)
    sigma_pressure = theta * (sigma_n / n - (1 + 1 / n))
    return factorization_pressure + divisor_pressure + sigma_pressure


def tnfr_is_prime(n: int) -> Tuple[bool, float]:
    dnfr = tnfr_delta_nfr(n)
    return (abs(dnfr) == 0.0, dnfr)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TNFR primality check using ΔNFR equations only")
    parser.add_argument("numbers", nargs="+", type=int, help="Integers to check")
    args = parser.parse_args(argv)

    header = f"{'n':>10}  {'TNFR_PRIME':>11}  {'ΔNFR':>14}"
    print(header)
    print("-" * len(header))
    for n in args.numbers:
        isp, dnfr = tnfr_is_prime(n)
        print(f"{n:10d}  {str(isp):>11}  {dnfr:14.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
