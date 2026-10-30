#!/usr/bin/env python3
"""Checkout entry point for the maintained TNFR arithmetic primality CLI.

This adapter requires the repository's engine dependencies. Arithmetic
factorization characterizes the prime zero set; it is not a physical nodal
derivation. Named helpers remain available for historical script consumers.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SOURCE_ROOT = str(Path(__file__).resolve().parents[1] / "src")
if _SOURCE_ROOT not in sys.path:
    sys.path.insert(0, _SOURCE_ROOT)

from tnfr.mathematics.arithmetic_pressure import big_omega
from tnfr.mathematics.arithmetic_pressure import (  # noqa: E402
    divisor_sum as _divisor_sum,
)
from tnfr.mathematics.arithmetic_pressure import num_divisors
from tnfr.tools.tnfr_is_prime_cli import (  # noqa: E402
    main,
    tnfr_delta_nfr,
    tnfr_is_prime,
)

__all__ = [
    "divisor_count",
    "divisor_sum",
    "prime_factor_count",
    "tnfr_delta_nfr",
    "tnfr_is_prime",
    "main",
]


def divisor_count(n: int) -> int:
    """Retain the script's n<=1 boundary and reuse shared integer arithmetic."""
    return (0 if n < 1 else 1) if n <= 1 else num_divisors(n)


def divisor_sum(n: int) -> int:
    """Retain the script's n<=1 boundary and reuse shared integer arithmetic."""
    return (0 if n < 1 else 1) if n <= 1 else _divisor_sum(n)


def prime_factor_count(n: int) -> int:
    """Count factors with multiplicity through the shared Omega owner."""
    return 0 if n < 2 else big_omega(n)


if __name__ == "__main__":
    raise SystemExit(main())
