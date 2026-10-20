"""Deprecated compatibility bridge for an externally supplied factorization lab.

The wrapper returns lab candidates, arithmetic telemetry and configured
acceptance reports. These are not a general factorization theorem or evidence
that factors emerge autonomously from the nodal equation. Grammar validation
of a proposed word does not verify its execution or arithmetic divisibility.

The in-repository lab is archived and is never discovered or imported by this
bridge. A separately supplied ``tnfr_factorization`` package may still serve
legacy callers, with a deprecation warning. Its results retain that package's
heuristic contracts and are not maintained engine certificates."""

from __future__ import annotations

import sys
import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import Any

__all__ = ["factorize", "SpectralAnalysisResult", "SpectralPaleyFactorizer"]

_RETIREMENT_GUIDANCE = (
    "The in-repository factorization lab is archived and is no longer "
    "loaded by the engine. Core arithmetic and spectral instruments are "
    "documented in theory/TNFR_NUMBER_THEORY.md. This deprecated bridge "
    "requires a separately supplied 'tnfr_factorization' package."
)


def _reject_archived_checkout() -> None:
    """Reject the archived source even when another consumer exposed its path."""
    from ..errors import TNFRUserError

    repository = Path(__file__).resolve().parents[3]
    archive = repository / "applications" / "factorization-lab"
    if not archive.is_dir():
        return
    loaded = sys.modules.get("tnfr_factorization")
    spec = (
        getattr(loaded, "__spec__", None)
        if loaded is not None
        else find_spec("tnfr_factorization")
    )
    origins = [
        getattr(loaded, "__file__", None),
        getattr(spec, "origin", None),
        *(getattr(spec, "submodule_search_locations", None) or ()),
    ]
    cached_child = sys.modules.get("tnfr_factorization.spectral_paley")
    origins.extend(
        (
            getattr(cached_child, "__file__", None),
            getattr(getattr(cached_child, "__spec__", None), "origin", None),
        )
    )
    for origin in origins:
        if origin and Path(origin).resolve().is_relative_to(archive.resolve()):
            raise TNFRUserError(_RETIREMENT_GUIDANCE)


def _load_spectral_paley() -> tuple[Any, Any]:
    """Load only an explicitly supplied package, without checkout discovery."""
    from ..errors import TNFRUserError

    _reject_archived_checkout()
    try:
        from tnfr_factorization.spectral_paley import (
            SpectralAnalysisResult,
            SpectralPaleyFactorizer,
        )

    except ModuleNotFoundError as exc:
        if exc.name not in {"tnfr_factorization", "tnfr_factorization.spectral_paley"}:
            raise
        raise TNFRUserError(_RETIREMENT_GUIDANCE) from exc
    warnings.warn(
        "tnfr.factorization is a deprecated bridge to an external legacy lab. "
        "Its heuristic results are not maintained factorization certificates.",
        DeprecationWarning,
        stacklevel=3,
    )
    return SpectralPaleyFactorizer, SpectralAnalysisResult


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
    """Run an externally supplied legacy analysis and return its full result.

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
    The repository lab is archived; this function does not load it. The
    compatibility wrapper reuses one external factorizer with its default node cap. It has
    ``trace_certificates`` rather than the lab wrapper's ``trace`` argument and
    no ``pure`` argument. ``TNFR_PURE_MODE`` selects a partial heuristic policy,
    not an arithmetic-free execution. See ``applications/factorization-lab/README.md``
    for the retirement boundary and known limitations of the archived source.
    """

    factorizer = _get_factorizer()
    return factorizer.analyze(
        n,
        modulus=modulus,
        trace_certificates=trace_certificates,
        certificate_dir=certificate_dir,
    )
