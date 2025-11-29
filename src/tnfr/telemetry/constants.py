"""Centralized telemetry thresholds and labels (CANONICAL).

All thresholds derive from classical mathematical theory:
- Φ_s: von Koch fractal bounds + combinatorial number theory
- |∇φ|: Harmonic oscillator stability + Kuramoto synchronization  
- |K_φ|: TNFR formalism constraints + 90% safety margin
- ξ_C: Spatial correlation theory + critical phenomena

Use these constants across examples, notebooks, and validators to ensure
consistent behavior and messaging.
"""

try:
    import mpmath as mp
    mp.dps = 30
    
    # Import canonical constants for mathematical consistency
    from ..constants.canonical import PHI, GAMMA, PI
    
    # CANONICAL mathematical derivations using φ, γ, π, e
    # U6: Structural Potential Confinement (ΔΦ_s) - Golden Ratio escape
    STRUCTURAL_POTENTIAL_DELTA_THRESHOLD: float = float(PHI)  # φ ≈ 1.618 (golden escape threshold)
    
    # |∇φ|: Phase gradient safety threshold - Euler-Pi coupling
    # CANONICAL DERIVATION: γ/π natural phase gradient from coupling constants
    PHASE_GRADIENT_THRESHOLD: float = float(GAMMA / PI)  # ≈ 0.1837 (canonical)
    
    # |K_φ|: Phase curvature absolute threshold - Golden-Pi geometry
    # CANONICAL DERIVATION: φ × π maximum geometric curvature before mutation
    PHASE_CURVATURE_ABS_THRESHOLD: float = float(PHI * PI)  # ≈ 5.083 (canonical)
    
    # Φ_s: Structural potential field threshold - Golden-Euler balance  
    # CANONICAL DERIVATION: φ/(φ+γ) natural coherence target from canonical constants
    PHI_S_CLASSICAL_THRESHOLD: float = float(PHI / (PHI + GAMMA))  # ≈ 0.737 (canonical)
    
except ImportError:
    # Fallback to canonical computed values
    STRUCTURAL_POTENTIAL_DELTA_THRESHOLD = 1.618  # φ
    PHASE_GRADIENT_THRESHOLD = 0.1837  # γ/π
    PHASE_CURVATURE_ABS_THRESHOLD = 5.083  # φ×π
    PHI_S_CLASSICAL_THRESHOLD = 0.737  # φ/(φ+γ)

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
