"""TNFR canonical factorization entry points.

This module exposes ``factorize`` as the supported interface for Paley-based
spectral factorizations. It wraps :class:`tnfr_factorization.spectral_paley.SpectralPaleyFactorizer`
so that downstream code always benefits from the official grammar validator and
self-optimization engines wired in the factorization lab.
"""

from __future__ import annotations

from pathlib import Path

import sys

from ..errors import TNFRUserError

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

_bootstrap_factorization_lab()

try:  # pragma: no cover - exercised indirectly through factorize()
    from tnfr_factorization.spectral_paley import (
        SpectralAnalysisResult,
        SpectralPaleyFactorizer,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - raised when extra missing
    raise TNFRUserError(
        "tnfr_factorization is not available. Install the tnfr-factorization package "
        "or keep the factorization-lab directory in your workspace."
    ) from exc

__all__ = ["factorize", "SpectralAnalysisResult", "SpectralPaleyFactorizer"]

_DEFAULT_FACTORIZER: SpectralPaleyFactorizer | None = None

def _get_factorizer() -> SpectralPaleyFactorizer:
    global _DEFAULT_FACTORIZER
    if _DEFAULT_FACTORIZER is None:
        _DEFAULT_FACTORIZER = SpectralPaleyFactorizer()
    return _DEFAULT_FACTORIZER

def factorize(
    n: int,
    *,
    modulus: int | None = None,
    trace_certificates: bool = False,
    certificate_dir: Path | None = None,
) -> SpectralAnalysisResult:
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
