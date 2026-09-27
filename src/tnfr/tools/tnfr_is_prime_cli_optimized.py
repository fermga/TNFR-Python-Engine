"""Compatibility entry point for the maintained arithmetic primality CLI.

Parsing, arithmetic, caches and optimized execution belong to
`tnfr_is_prime_cli`. The old independent benchmark is retired; its compatibility
call now returns the shared optimized benchmark schema.
"""

from __future__ import annotations

from . import tnfr_is_prime_cli as _owner
from .tnfr_is_prime_cli import main, tnfr_delta_nfr, tnfr_delta_nfr_cached

__all__ = [
    "main",
    "tnfr_delta_nfr",
    "tnfr_delta_nfr_cached",
    "tnfr_is_prime",
    "benchmark_basic",
]


def tnfr_is_prime(n: int, *, use_cached: bool = False) -> tuple[bool, float]:
    """Preserve the basic decision API; its shared owner already caches inputs."""
    return _owner.tnfr_is_prime(n)


def benchmark_basic(max_n: int = 10000, sample_size: int = 100) -> dict:
    """Return the maintained benchmark schema through the historical entry."""
    return _owner.benchmark_optimization(max_n, sample_size)


if __name__ == "__main__":
    raise SystemExit(main())
