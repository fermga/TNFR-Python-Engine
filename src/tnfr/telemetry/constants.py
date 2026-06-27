"""Centralized telemetry thresholds and labels.

Threshold provenance (audit 2026 — only the π phase-wrap is genuinely derived):
- Φ_s: per-node threshold π/4 ≈ 0.785 (quarter phase-wrap)
- |∇φ|: kinematic bound is π (phase wrap); the early-warning level ≈ 0.184
  is a heuristic only, NOT derived (measured sync-onset ≈ 0.29)
- |K_φ|: π phase-wrap bound with 90% safety margin (genuine)
- ξ_C: scale set by the spectral gap (ξ_C ∝ 1/√λ₂)

Use these constants across examples, notebooks, and validators to ensure
consistent behavior and messaging.
"""

from ..constants.canonical import PHASE_GRADIENT_THRESHOLD_CANONICAL, PI, U6_STRUCTURAL_POTENTIAL_LIMIT

# U6: Structural Potential Confinement (ΔΦ_s) — half phase-wrap bound π/2
STRUCTURAL_POTENTIAL_DELTA_THRESHOLD: float = float(
    U6_STRUCTURAL_POTENTIAL_LIMIT
)  # π/2 ≈ 1.571 (structural confinement bound)

# |∇φ|: Phase gradient early-warning (heuristic, audit 2026: not derived; bound is π)
PHASE_GRADIENT_THRESHOLD: float = float(
    PHASE_GRADIENT_THRESHOLD_CANONICAL
)  # ≈ 0.196 (π/16, heuristic)

# |K_φ|: Phase curvature safety threshold. Audit 2026: |K_φ| ≤ π by phase wrap,
# so the genuine threshold is 0.9π ≈ 2.827; the earlier  ≈ 5.083 was
# NON-PHYSICAL (unreachable, so any check using it was a no-op) — corrected.
PHASE_CURVATURE_ABS_THRESHOLD: float = float(0.9 * PI)  # ≈ 2.827 (phase wrap)

# Φ_s: Structural potential threshold (operational; not derived)
PHI_S_CLASSICAL_THRESHOLD: float = float(0.75)  # ≈ 0.737 (operational)

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
