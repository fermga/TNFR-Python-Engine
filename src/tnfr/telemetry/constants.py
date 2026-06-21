"""Centralized telemetry thresholds and labels.

Threshold provenance (audit 2026 — only the π phase-wrap is genuinely derived):
- Φ_s: empirical per-node threshold (0.7711; no closed-form derivation)
- |∇φ|: kinematic bound is π (phase wrap); γ/π ≈ 0.184 is a heuristic
  early-warning level only, NOT derived (measured sync-onset ≈ 0.29)
- |K_φ|: π phase-wrap bound with 90% safety margin (genuine)
- ξ_C: scale set by the spectral gap (ξ_C ∝ 1/√λ₂), not base e

Use these constants across examples, notebooks, and validators to ensure
consistent behavior and messaging.
"""

from ..constants.canonical import (
    PHI,
    PI,
    PHASE_GRADIENT_THRESHOLD_CANONICAL,
    PHI_GAMMA_NORMALIZED,
)

# CANONICAL mathematical constants using φ, γ, π, e (notational convention)
# U6: Structural Potential Confinement (ΔΦ_s) - empirical golden-ratio escape
STRUCTURAL_POTENTIAL_DELTA_THRESHOLD: float = float(PHI)  # φ ≈ 1.618 (empirical escape threshold)

# |∇φ|: Phase gradient early-warning (heuristic, audit 2026: not derived; bound is π)
PHASE_GRADIENT_THRESHOLD: float = float(PHASE_GRADIENT_THRESHOLD_CANONICAL)  # ≈ 0.1837 (heuristic)

# |K_φ|: Phase curvature safety threshold. Audit 2026: |K_φ| ≤ π by phase wrap,
# so the genuine threshold is 0.9π ≈ 2.827; the earlier φ×π ≈ 5.083 was
# NON-PHYSICAL (unreachable, so any check using it was a no-op) — corrected.
PHASE_CURVATURE_ABS_THRESHOLD: float = float(0.9 * PI)  # ≈ 2.827 (phase wrap)

# Φ_s: Structural potential threshold - notational φ/(φ+γ) (not derived)
PHI_S_CLASSICAL_THRESHOLD: float = float(PHI_GAMMA_NORMALIZED)  # ≈ 0.737 (notational)

# ξ_C locality gate description for documentation/UI (read-only guidance)
XI_C_LOCALITY_RULE: str = "local regime if ξ_C < mean_path_length"

# Human-facing labels to keep UIs and reports consistent
TELEMETRY_LABELS = {
    "phi_s": "Φ_s (structural potential)",
    "dphi_s": "ΔΦ_s (drift)",
    "grad": "|∇φ| (phase gradient)",
    "kphi": "|K_φ| (phase curvature)",
    "xi_c": "ξ_C (coherence length)",
}
