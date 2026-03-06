"""TNFR canonical factorization entry points.

This module exposes ``factorize`` as the supported interface for Paley-based
spectral factorizations. It wraps :class:`tnfr_factorization.spectral_paley.SpectralPaleyFactorizer`
so that downstream code always benefits from the official grammar validator and
self-optimization engines wired in the factorization lab.

The ``tnfr_factorization`` external package is an **optional** dependency.
Importing this module always succeeds; errors are deferred until a function
is actually called without the dependency installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import sys

__all__ = ["factorize", "SpectralAnalysisResult", "SpectralPaleyFactorizer"]


def _bootstrap_factorization_lab() -> None:
    """Ensure the tnfr_factorization package is importable.

    In development checkouts the factorization lab lives in the sibling
    ``factorization-lab`` directory. When the standalone ``tnfr-factorization``
    package is installed, this function becomes a no-op.
    """

    if "tnfr_factorization" in sys.modules:
        return

    lab_root = Path(__file__).resolve().parents[3] / "factorization-lab"
    if lab_root.exists() and str(lab_root) not in sys.path:
        sys.path.insert(0, str(lab_root))


def _load_spectral_paley() -> tuple[Any, Any]:
    """Lazy-load the spectral_paley module, raising a clear error if missing."""
    from ..errors import TNFRUserError

    _bootstrap_factorization_lab()
    try:
        from tnfr_factorization.spectral_paley import (
            SpectralAnalysisResult,
            SpectralPaleyFactorizer,
        )
        return SpectralPaleyFactorizer, SpectralAnalysisResult
    except ModuleNotFoundError as exc:
        raise TNFRUserError(
            "tnfr_factorization is not available. Install the tnfr-factorization "
            "package or keep the factorization-lab directory in your workspace."
        ) from exc


# Lazy accessors for type re-exports
def __getattr__(name: str) -> Any:
    if name in ("SpectralPaleyFactorizer", "SpectralAnalysisResult"):
        _Factorizer, _Result = _load_spectral_paley()
        globals()["SpectralPaleyFactorizer"] = _Factorizer
        globals()["SpectralAnalysisResult"] = _Result
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_DEFAULT_FACTORIZER: Any = None


def _get_factorizer() -> Any:
    global _DEFAULT_FACTORIZER
    if _DEFAULT_FACTORIZER is None:
        _Factorizer, _ = _load_spectral_paley()
        _DEFAULT_FACTORIZER = _Factorizer()
    return _DEFAULT_FACTORIZER


def factorize(
    n: int,
    *,
    modulus: int | None = None,
    trace_certificates: bool = False,
    certificate_dir: Path | None = None,
) -> Any:
    """Run canonical Paley spectral factorization.

    Parameters
    ----------
    n:
        Composite candidate. Must be ``> 1``.
    modulus:
        Optional Paley modulus override. When omitted the factorizer derives
        the closest admissible modulus automatically.
    trace_certificates:
        If ``True`` the factorizer emits operator certificates that already
        include U1-U6 grammar validation metadata.
    certificate_dir:
        Optional directory for certificate emission.

    Returns
    -------
    SpectralAnalysisResult
        Complete spectral telemetry together with candidate factors and
        optimization metadata.
    """

    factorizer = _get_factorizer()
    return factorizer.analyze(
        n,
        modulus=modulus,
        trace_certificates=trace_certificates,
        certificate_dir=certificate_dir,
    )
