"""Centralized telemetry thresholds and labels.

Threshold provenance (audit 2026):
- Φ_s: selected per-node warning π/4 and U6 drift policy π/2
- |∇φ|: selected early-warning policy π/16 ≈ 0.196; exact bound π;
  measured synchronization onset ≈ 0.29 and σ-dependent
- |K_φ|: selected 0.9π warning margin; exact bound π
- ξ_C: fitted length and spectral fallback retain their distinct scope

Use these constants across examples, notebooks, and validators to ensure
consistent behavior and messaging.
"""

from ..constants.canonical import (
    GRAD_PHI_CANONICAL_THRESHOLD,
    K_PHI_CANONICAL_THRESHOLD,
    PHI_S_VON_KOCH_THRESHOLD,
    U6_STRUCTURAL_POTENTIAL_LIMIT,
)

# U6: selected structural-potential drift policy (ΔΦ_s), π/2.
STRUCTURAL_POTENTIAL_DELTA_THRESHOLD: float = float(
    U6_STRUCTURAL_POTENTIAL_LIMIT
)  # π/2 ≈ 1.571 (selected policy, not a graph-independent bound)

# |∇φ|: Phase gradient early-warning (heuristic, audit 2026: not derived; bound is π)
PHASE_GRADIENT_THRESHOLD: float = float(
    GRAD_PHI_CANONICAL_THRESHOLD
)  # ≈ 0.196 (π/16, heuristic)

# |K_φ|: selected warning margin. The exact wrapped-angle bound is π; the
# earlier ≈5.083 threshold was unreachable and made the check a no-op.
PHASE_CURVATURE_ABS_THRESHOLD: float = float(
    K_PHI_CANONICAL_THRESHOLD
)  # 0.9π ≈ 2.827 (selected warning margin)

# Φ_s: legacy public name for the selected per-node potential warning policy.
PHI_S_CLASSICAL_THRESHOLD: float = float(
    PHI_S_VON_KOCH_THRESHOLD
)  # π/4 ≈ 0.785 (selected policy)

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
