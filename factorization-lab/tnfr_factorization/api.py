"""Public high-level TNFR factorization API.

All factor recovery emerges from nodal dynamics: Laplacian spectrum -> tetrad proxies ->
partition nodal decoding -> structural verification. This wrapper exposes a stable
entry-point without requiring direct interaction with the lower level factorizer.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from .spectral_paley import SpectralAnalysisResult, SpectralPaleyFactorizer

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
    telemetry: Dict[str, Any]
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

    pure_mode_active = (
        os.getenv("TNFR_PURE_MODE", "").lower() in {"1", "true", "yes", "on"}
        if pure is None
        else pure
    )

    telemetry: Dict[str, Any] = {
        # Structural Field Tetrad (§7, TNFR_NUMBER_THEORY.md)
        "phi_s": analysis.phi_s,
        "phase_gradient": analysis.phase_gradient,
        "phase_curvature": analysis.phase_curvature,
        "coherence_length": analysis.coherence_length,
        "coherence_score": analysis.coherence_score,
        # Nodal equation components (§5-6)
        "delta_nfr": analysis.arithmetic_delta_nfr,
        "epi": analysis.arithmetic_epi,
        "nu_f": analysis.arithmetic_nu_f,
        "local_coherence": analysis.arithmetic_local_coherence,
        # Pressure decomposition (§6, component_breakdown)
        "pressure_components": analysis.arithmetic_components,
        # Conservation proxies (Noether charge Q = Φ_s + K_φ, Lyapunov E)
        "noether_charge_proxy": analysis.phi_s + analysis.phase_curvature,
        "energy_proxy": 0.5
        * (
            analysis.phi_s**2 + analysis.phase_gradient**2 + analysis.phase_curvature**2
        ),
        # Dual-lever analysis (§8)
        "dual_lever": analysis.dual_lever_analysis,
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
