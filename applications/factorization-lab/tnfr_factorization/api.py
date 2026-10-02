"""High-level wrapper for the experimental factorization lab.

The pipeline combines a supplied arithmetic graph, spectral heuristics,
classical arithmetic telemetry and optional partition workflows. It does not
derive autonomous factor recovery from the nodal equation. Candidate and
structural-acceptance labels require separate arithmetic verification."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from .spectral_paley import SpectralAnalysisResult, SpectralPaleyFactorizer

__all__ = ["FactorizationResult", "factorize"]


@dataclass
class FactorizationResult:
    """User-facing candidate analysis and diagnostic records.

    ``tnfr_certified_factors`` retains its compatibility name for heuristic
    acceptance; its verifier records divisibility separately and does not use
    that field in the final acceptance condition. ``pure_mode`` identifies a
    partial policy choice, not absence of arithmetic throughout the pipeline.
    Telemetry names include lab proxies, not necessarily canonical graph fields."""

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
    """Analyze integer ``n`` with spectral and arithmetic lab heuristics.

    Parameters
    ----------
    n: int
        Integer greater than one.
    pure: bool | None
        Set the temporary ``TNFR_PURE_MODE`` policy, or inherit the environment
        when None. True skips arithmetic-hint injection and gcd refinement in
        the initial seed stage; arithmetic telemetry, divisibility-based size
        hints and the empty-candidate fallback remain in the full pipeline.
    trace: bool
        Emit analysis, partition and optional workflow records. Their legacy
        certificate names do not establish an executed factorization proof.
    max_nodes: int | None
        Maximum graph size. None is passed through as no cap, unlike the
        low-level factorizer constructor's omitted-argument default of 4097.
    modulus: int | None
        Optional graph modulus; odd values 1 modulo 4 are the intended domain.
        The builder checks the lower bound 5, not all mathematical hypotheses
        of classical prime-modulus Paley graph results.

    Notes
    -----
    The temporary environment override is restored on return. It is process-wide
    while the call runs, so concurrent callers must coordinate different modes.
    Verify each returned proper factor arithmetically. ``trace=False`` suppresses
    explicit certificate export, not necessarily all optional diagnostic output."""
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
        # Lab spectral proxies; these are not the canonical nodewise tetrad.
        "phi_s": analysis.phi_s,
        "phase_gradient": analysis.phase_gradient,
        "phase_curvature": analysis.phase_curvature,
        "coherence_length": analysis.coherence_length,
        "coherence_score": analysis.coherence_score,
        # Static arithmetic readouts; factor/divisor statistics are supplied.
        "delta_nfr": analysis.arithmetic_delta_nfr,
        "epi": analysis.arithmetic_epi,
        "nu_f": analysis.arithmetic_nu_f,
        "local_coherence": analysis.arithmetic_local_coherence,
        # Pressure decomposition (§6, component_breakdown)
        "pressure_components": analysis.arithmetic_components,
        # Descriptive combinations only; no conservation or Lyapunov proof.
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
