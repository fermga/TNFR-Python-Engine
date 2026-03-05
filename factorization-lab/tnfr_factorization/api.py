"""Public high-level TNFR factorization API.

All factor recovery emerges from nodal dynamics: Laplacian spectrum -> tetrad proxies ->
partition nodal decoding -> structural verification. This wrapper exposes a stable
entry-point without requiring direct interaction with the lower level factorizer.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import os

from .spectral_paley import SpectralPaleyFactorizer, SpectralAnalysisResult

__all__ = ["FactorizationResult", "factorize"]


@dataclass
class FactorizationResult:
    """User-facing factorization outcome.

    Fields consolidate spectral + TNFR verification artifacts. Arithmetic checks are
    optional; in pure mode the candidate list is nodal / structural only.
    """
    n: int
    modulus: int
    candidate_factors: List[int]
    tnfr_certified_factors: List[int]
    factor_signature: Dict[str, Any] | None
    composite_signature: Dict[str, Any] | None
    pure_mode: bool
    notes: str
    telemetry: Dict[str, float]
    certificate_path: Optional[str] = None
    partition_manifest_path: Optional[str] = None
    tnfr_verification: Dict[str, Any] | None = None
    failure_diagnostics: Dict[str, Any] | None = None

    def to_mapping(self) -> Dict[str, Any]:
        payload = asdict(self)
        return payload


def factorize(
    n: int,
    *,
    pure: bool | None = None,
    trace: bool = False,
    max_nodes: int | None = None,
    modulus: int | None = None,
) -> FactorizationResult:
    """Factorize integer ``n`` using TNFR spectral-nodal dynamics.

    Parameters
    ----------
    n: int
        Integer to factor (>1).
    pure: bool | None
        If True, enable pure TNFR mode (no gcd refinement). If False, retain
        assisted arithmetic hints. If None, use existing environment setting.
    trace: bool
        Whether to emit operator certificate artifacts.
    max_nodes: int | None
        Override maximum graph nodes for Paley construction.
    modulus: int | None
        Optional explicit Paley modulus (must be 1 mod 4 and odd).
    """
    if n <= 1:
        raise ValueError("n must be > 1")

    previous = os.getenv("TNFR_PURE_MODE")
    if pure is not None:
        os.environ["TNFR_PURE_MODE"] = "1" if pure else "0"
    try:
        factorizer = SpectralPaleyFactorizer(max_nodes=max_nodes)
        analysis: SpectralAnalysisResult = factorizer.analyze(
            n,
            modulus=modulus,
            trace_certificates=trace,
        )
    finally:
        if pure is not None:
            # Restore previous state to avoid side-effects.
            if previous is None:
                os.environ.pop("TNFR_PURE_MODE", None)
            else:
                os.environ["TNFR_PURE_MODE"] = previous

    pure_mode_active = os.getenv("TNFR_PURE_MODE", "").lower() in {"1", "true", "yes", "on"} if pure is None else pure

    telemetry = {
        "phi_s": analysis.phi_s,
        "phase_gradient": analysis.phase_gradient,
        "phase_curvature": analysis.phase_curvature,
        "coherence_length": analysis.coherence_length,
        "coherence_score": analysis.coherence_score,
        "delta_nfr": analysis.arithmetic_delta_nfr,
    }

    result = FactorizationResult(
        n=analysis.n,
        modulus=analysis.modulus,
        candidate_factors=list(analysis.candidate_factors),
        tnfr_certified_factors=analysis.tnfr_certified_factors or [],
        factor_signature=analysis.tnfr_factor_signature,
        composite_signature=analysis.tnfr_composite_signature,
        pure_mode=pure_mode_active,
        notes=analysis.notes,
        telemetry=telemetry,
        certificate_path=analysis.certificate_path,
        partition_manifest_path=analysis.partition_manifest_path,
        tnfr_verification=analysis.tnfr_verification,
        failure_diagnostics=analysis.failure_diagnostics,
    )
    return result
