"""Centralized telemetry thresholds and labels (CANONICAL).

Use these constants across examples, notebooks, and validators to ensure
consistent behavior and messaging.
"""

# U6: Structural Potential Confinement (ΔΦ_s)
STRUCTURAL_POTENTIAL_DELTA_THRESHOLD: float = 2.0

# |∇φ|: Phase gradient safety threshold (local stress)
PHASE_GRADIENT_THRESHOLD: float = 0.38

# |K_φ|: Phase curvature absolute threshold (geometric confinement)
PHASE_CURVATURE_ABS_THRESHOLD: float = 3.0

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
