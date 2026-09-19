"""Optional entry points for the experimental spectral factorization lab.

The wrapper returns lab candidates, arithmetic telemetry and configured
acceptance reports. These are not a general factorization theorem or evidence
that factors emerge autonomously from the nodal equation. Grammar validation
of a proposed word does not verify its execution or arithmetic divisibility.

The ``tnfr_factorization`` package is optional; a source checkout can load the
sibling ``factorization-lab`` directory. Missing dependencies are reported when
the factorizer is first requested."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

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
    """Run the experimental lab analysis and return its full result.

    Parameters
    ----------
    n:
        Integer candidate greater than one.
    modulus:
        Optional graph modulus. The automatic route chooses an integer at least
        n and 5 that is 1 modulo 4; it does not require a prime modulus.
    trace_certificates:
        Emit analysis records with proposed operator words and grammar metadata.
        These records alone certify neither operator execution nor factors.
    certificate_dir:
        Optional directory for the analysis records.

    Returns
    -------
    SpectralAnalysisResult
        Spectral proxies, arithmetic telemetry, candidates and heuristic reports.
        Check ``1 < d < n and n % d == 0`` for every claimed proper factor.

    Notes
    -----
    This wrapper reuses one factorizer with its class default node cap. It has
    ``trace_certificates`` rather than the lab wrapper's ``trace`` argument and
    no ``pure`` argument. ``TNFR_PURE_MODE`` selects a partial heuristic policy,
    not an arithmetic-free execution; see ``factorization-lab/README.md``."""

    factorizer = _get_factorizer()
    return factorizer.analyze(
        n,
        modulus=modulus,
        trace_certificates=trace_certificates,
        certificate_dir=certificate_dir,
    )
