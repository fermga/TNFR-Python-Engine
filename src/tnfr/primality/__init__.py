"""TNFR canonical primality entry points.

This module exposes canonical primality utilities through a stable interface:
``is_prime``, ``delta_nfr``, ``component_breakdown``, ``structural_triad``, and
``analyze``.

The ``tnfr_primality`` external package is an **optional** dependency.
Importing this module always succeeds; errors are deferred until a function
is actually called without the dependency installed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

__all__ = [
    "is_prime",
    "delta_nfr",
    "component_breakdown",
    "structural_triad",
    "analyze",
]


def _bootstrap_primality_test() -> None:
    """Ensure the tnfr_primality package is importable.

    In development checkouts the primality module lives in the sibling
    ``primality-test`` directory. When the standalone ``tnfr-primality``
    package is installed, this function becomes a no-op.
    """
    if "tnfr_primality" in sys.modules:
        return

    test_root = Path(__file__).resolve().parents[3] / "primality-test"
    if test_root.exists() and str(test_root) not in sys.path:
        sys.path.insert(0, str(test_root))


def _load_core() -> tuple[Any, Any, Any, Any]:
    """Lazy-load core primality functions, raising clear errors if missing."""
    from ..errors import TNFRUserError

    _bootstrap_primality_test()
    try:
        from tnfr_primality import (
            tnfr_component_breakdown,
            tnfr_delta_nfr,
            tnfr_is_prime,
            tnfr_structural_triad,
        )

        return (
            tnfr_is_prime,
            tnfr_delta_nfr,
            tnfr_component_breakdown,
            tnfr_structural_triad,
        )
    except ModuleNotFoundError as exc:
        raise TNFRUserError(
            "tnfr_primality is not available. Install the tnfr-primality "
            "package or keep the primality-test directory in your workspace."
        ) from exc


def delta_nfr(n: int) -> float:
    """Compute arithmetic TNFR pressure $\\Delta NFR(n)$."""
    _, tnfr_delta_nfr, _, _ = _load_core()
    return float(tnfr_delta_nfr(n))


def is_prime(n: int, *, tolerance: float = 1e-10) -> tuple[bool, float]:
    """Run TNFR primality test and return ``(is_prime, delta_nfr)``."""
    tnfr_is_prime, _, _, _ = _load_core()
    result, dnfr = tnfr_is_prime(n, tolerance=tolerance)
    return bool(result), float(dnfr)


def component_breakdown(n: int) -> dict[str, Any]:
    """Return per-component $\\Delta NFR$ decomposition."""
    _, _, tnfr_component_breakdown, _ = _load_core()
    return dict(tnfr_component_breakdown(n))


def structural_triad(n: int) -> dict[str, Any]:
    """Return TNFR structural triad {EPI, vf, delta_nfr, local_coherence}."""
    _, _, _, tnfr_structural_triad = _load_core()
    return dict(tnfr_structural_triad(n))


def analyze(n: int, *, tolerance: float = 1e-10) -> dict[str, Any]:
    """Return complete canonical primality analysis payload for SDK/reporting."""
    prime, dnfr = is_prime(n, tolerance=tolerance)
    breakdown = component_breakdown(n)
    triad = structural_triad(n)
    return {
        "n": int(n),
        "is_prime": bool(prime),
        "delta_nfr": float(dnfr),
        "components": breakdown,
        "triad": triad,
        "tolerance": float(tolerance),
    }
